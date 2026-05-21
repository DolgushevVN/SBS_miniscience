import numpy as np
import scipy.constants as const
from dolfinx import fem
from dolfinx.mesh import create_submesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import ufl
import basix.ufl

from utils.mesher import create_waveguide_mesh
from utils.materials import Material
from physics.em_solver import EMSolver
from physics.elastic_solver import ElasticSolver
from physics.gain_calc import BrillouinGainCalculator

def main():
    print("=== Начало расчета ВРМБ (Кремний в воздухе) ===")
    
    w_sim, h_sim = 1.5, 1.5
    w_wg, h_wg = 0.45, 0.315
    res_wg = 0.0075   
    res_clad = 0.0075 

    # Точный кубический анизотропный тензор упругости кремния (Voigt 6x6)
    C_silicon_cubic = np.array([
        [165.7,  63.9,  63.9,   0.0,   0.0,   0.0],
        [ 63.9, 165.7,  63.9,   0.0,   0.0,   0.0],
        [ 63.9,  63.9, 165.7,   0.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,  79.6,   0.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,  79.6,   0.0],
        [  0.0,   0.0,   0.0,   0.0,   0.0,  79.6]
    ])

    mat_si = Material("Silicon", rho=2.328, M=165.7, G=79.6, n_ref=3.48, p11=-0.09, p12=0.017)
    mat_air = Material("Air", rho=1.2e-3, M=0.0, G=0.0, n_ref=1.0, p11=0.0, p12=0.0)
    materials = {1: mat_air, 2: mat_si}

    print("Генерация сетки...")
    domain, cell_tags, facet_tags = create_waveguide_mesh(w_sim, h_sim, w_wg, h_wg, res_wg, res_clad)

    # 1. Считаем оптику
    lam = 1.55
    dlam = 0.001
    
    print(f"Оптический шаг 1 (длина волны {lam + dlam} мкм)...")
    em_solver2 = EMSolver(domain, cell_tags, materials, lam + dlam)
    _, n_eff2 = em_solver2.solve(guess_neff=2.5, n_modes=1)
    
    print(f"Оптический шаг 2 (длина волны {lam} мкм)...")
    em_solver = EMSolver(domain, cell_tags, materials, lam)
    E_opt, n_eff = em_solver.solve(guess_neff=2.5, n_modes=1)
    
    ng = n_eff - lam * (n_eff2 - n_eff) / dlam
    print(f"-> Эффективный индекс n_eff = {n_eff:.5f}")
    print(f"-> Групповой индекс n_g = {ng:.5f}")
    
    power_opt = em_solver.calculate_power(E_opt, n_eff, ng)

    # 2. Выделение субсетки кремния для упругой задачи
    print("Выделение упругого домена (кремний)...")
    core_cells = cell_tags.find(2)
    submesh, entity_map, _, _ = create_submesh(domain, domain.topology.dim, core_cells)

    # 3. Решаем механику на субсетке
    q_b = 0.0 
    print("Поиск упругой (акустической) моды на субсетке...")
    elastic_solver = ElasticSolver(submesh, mat_si.rho, C_silicon_cubic, q_b)
    U_mech, freq_mech = elastic_solver.solve(guess_freq=8.2, n_modes=1)
    
    print(f"-> Частота упругой моды = {freq_mech:.4f} ГГц")
    power_mech = elastic_solver.calculate_power(U_mech, freq_mech)

    # 4. Проекция оптического поля в 3D Lagrange
    print("Проекция и перенос оптического поля на субсетку...")
    V_lag3 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    Et, Ez = ufl.split(E_opt)
    E_vec_ufl = ufl.as_vector([Et[0], Et[1], 1j * Ez]) # Стабильный перенос
    
    u_p = ufl.TrialFunction(V_lag3)
    v_p = ufl.TestFunction(V_lag3)
    a_proj = ufl.inner(u_p, v_p) * ufl.dx
    L_proj = ufl.inner(E_vec_ufl, v_p) * ufl.dx
    
    A_proj = assemble_matrix(fem.form(a_proj))
    A_proj.assemble()
    b_proj = assemble_vector(fem.form(L_proj))
    b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A_proj)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    E_lag3 = fem.Function(V_lag3)
    solver.solve(b_proj, E_lag3.x.petsc_vec)
    E_lag3.x.scatter_forward()
    
    # Интерполируем на субсетку
    V_sub3 = fem.functionspace(submesh, ("Lagrange", 1, (3,)))
    E_opt_sub = fem.Function(V_sub3)
    E_opt_sub.interpolate(E_lag3)

    # 5. Расчет усиления на субсетке
    print("Вычисление усиления ВРМБ (Gain)...")
    omega_opt = 2.0 * np.pi * const.c / (lam * 1e-6)
    omega_mech = freq_mech * 2.0 * np.pi * 1e9
    
    gain_calc = BrillouinGainCalculator(submesh)
    gain_bdr, gain_bulk, gain_total = gain_calc.calculate_total_gain(
        E_opt_sub, U_mech, omega_opt, omega_mech, 
        Q_mech=1000, power_opt=power_opt, P_mech=power_mech, 
        mat_core=mat_si, mat_clad=mat_air
    )
    
    print(f"\n=== Расчет окончен! ===")
    print(f"Radiation pressure gain (на границе): {gain_bdr:.1f} W^-1 m^-1")
    print(f"Bulk electrostriction gain (в объеме): {gain_bulk:.1f} W^-1 m^-1")
    print(f"Total SBS Gain (Полное усиление): {gain_total:.1f} W^-1 m^-1")

if __name__ == "__main__":
    main()


