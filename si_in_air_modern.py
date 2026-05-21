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
    w_wg, h_wg = 0.45, 0.23 
    
    # КРИТИЧЕСКИЙ ФИКС: Делаем сетку почти в два раза тоньше для точного поиска деформационных мод
    res_wg = 0.0005   
    res_clad = 0.001 

    C_silicon_cubic = np.array([
        [164.0,  64.0,  64.0,   0.0,   0.0,   0.0],
        [ 64.0, 164.0,  64.0,   0.0,   0.0,   0.0],
        [ 64.0,  64.0, 164.0,   0.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,  79.0,   0.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,  79.0,   0.0],
        [  0.0,   0.0,   0.0,   0.0,   0.0,  79.0]
    ])

    mat_si = Material("Silicon", rho=2.328, M=164.0, G=79.0, n_ref=3.48, p11=-0.09, p12=0.017)
    mat_air = Material("Air", rho=1.2e-3, M=0.0, G=0.0, n_ref=1.0, p11=0.0, p12=0.0)
    materials = {1: mat_air, 2: mat_si}

    print("Генерация высокоточной сетки...")
    domain, cell_tags, facet_tags = create_waveguide_mesh(w_sim, h_sim, w_wg, h_wg, res_wg, res_clad)

    lam = 1.55
    dlam = 0.001
    
    em_solver2 = EMSolver(domain, cell_tags, materials, lam + dlam)
    _, n_eff2 = em_solver2.solve(guess_neff=2.5, n_modes=1)
    
    em_solver = EMSolver(domain, cell_tags, materials, lam)
    E_opt, n_eff = em_solver.solve(guess_neff=2.5, n_modes=1)
    
    ng = n_eff - lam * (n_eff2 - n_eff) / dlam
    print(f"-> Эффективный индекс n_eff = {n_eff:.5f}")
    power_opt = em_solver.calculate_power(E_opt, n_eff, ng)

    print("Выделение упругого домена (кремний)...")
    core_cells = cell_tags.find(2)
    submesh, entity_map, _, _ = create_submesh(domain, domain.topology.dim, core_cells)

    q_b = 0.0 
    # Смещаем центр поиска ближе к 9.3 ГГц и ищем до 20 мод
    print("Поиск акустических мод на мелкой сетке (вычисляем 20 штук)...")
    elastic_solver = ElasticSolver(submesh, mat_si.rho, C_silicon_cubic, q_b)
    mech_modes = elastic_solver.solve(guess_freq=9.3, n_modes=20)

    print("Прямая интерполяция оптического поля на субсетку...")
    V_sub_t = fem.functionspace(submesh, ("Lagrange", 1, (2,)))
    V_sub_z = fem.functionspace(submesh, ("Lagrange", 1))
    
    E_t_sub = fem.Function(V_sub_t)
    E_t_sub.interpolate(E_opt.sub(0))
    
    E_z_sub = fem.Function(V_sub_z)
    E_z_sub.interpolate(E_opt.sub(1))

    print("\n=== РАСЧЕТ УСИЛЕНИЯ ===")
    omega_opt = 2.0 * np.pi * const.c / (lam * 1e-6)
    gamma_val = (2.0 * np.pi / lam) * n_eff
    
    gain_calc = BrillouinGainCalculator(submesh)
    mech_modes.sort(key=lambda x: x[1])
    
    for i, (U_mech, freq_mech) in enumerate(mech_modes):
        if freq_mech < 1.0:
            continue
            
        omega_mech = freq_mech * 2.0 * np.pi * 1e9
        power_mech = elastic_solver.calculate_power(U_mech, freq_mech)
        
        if power_mech == 0.0:
            continue
            
        gain_rp, gain_bulk, gain_bes, gain_total = gain_calc.calculate_total_gain(
            E_t_sub, E_z_sub, U_mech, omega_opt, omega_mech, 
            Q_mech=249, power_opt=power_opt, P_mech=power_mech,
            mat_core=mat_si, mat_clad=mat_air, gamma_val=gamma_val
        )
        
        if gain_total > 1.0:
            print(f"\nМода {i+1}: {freq_mech:.4f} ГГц  <--- НАЙДЕНА АКТИВНАЯ МОДА!")
            print(f"  --> Радиационное давление (RP):  {gain_rp:6.1f} W^-1 m^-1")
            print(f"  --> Объемная электрострикция:    {gain_bulk:6.1f} W^-1 m^-1")
            print(f"  --> Граничная электрострикция:   {gain_bes:6.1f} W^-1 m^-1")
            print(f"  --> ПОЛНОЕ УСИЛЕНИЕ (Gain):      {gain_total:6.1f} W^-1 m^-1")
        else:
            print(f"Мода {i+1}: {freq_mech:.4f} ГГц  (Ортогональна, Gain = {gain_total:.2e} W^-1 m^-1)")

    print(f"\n=== Расчет окончен! ===")

if __name__ == "__main__":
    main()


