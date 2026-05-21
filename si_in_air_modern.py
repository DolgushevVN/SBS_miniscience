import numpy as np
import scipy.constants as const
from dolfinx import fem
from dolfinx.mesh import create_submesh
import ufl

from utils.mesher import create_waveguide_mesh
from utils.materials import Material
from physics.em_solver import EMSolver
from physics.elastic_solver import ElasticSolver
from physics.gain_calc import BrillouinGainCalculator

def main():
    print("=== Начало расчета ВРМБ (Кремний в воздухе) ===")
    w_sim, h_sim = 1.5, 1.5
    w_wg, h_wg = 0.45, 0.23 
    res_wg, res_clad = 0.005, 0.01 

    C_si = np.array([
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

    print("Генерация сетки...")
    domain, cell_tags, facet_tags = create_waveguide_mesh(w_sim, h_sim, w_wg, h_wg, res_wg, res_clad)

    lam, dlam = 1.55, 0.001
    em_solver2 = EMSolver(domain, cell_tags, materials, lam + dlam)
    _, n_eff2 = em_solver2.solve(guess_neff=2.5, n_modes=1)
    
    em_solver = EMSolver(domain, cell_tags, materials, lam)
    E_opt, n_eff = em_solver.solve(guess_neff=2.5, n_modes=1)
    ng = n_eff - lam * (n_eff2 - n_eff) / dlam
    print(f"-> n_eff = {n_eff:.5f}, ng = {ng:.5f}")
    
    power_opt = em_solver.calculate_power(E_opt, n_eff, ng)
    print(f"-> Оптическая мощность P_opt = {power_opt:.2e} Вт")

    core_cells = cell_tags.find(2)
    submesh, entity_map, _, _ = create_submesh(domain, domain.topology.dim, core_cells)

    print("Безопасная Expression-интерполяция оптики на субсетку...")
    V_dg_t_parent = fem.functionspace(domain, ("DG", 1, (2,)))
    V_dg_z_parent = fem.functionspace(domain, ("DG", 1))
    
    # Решаем проблему TypeError (используем свойство вместо метода)
    pts_t = V_dg_t_parent.element.interpolation_points
    pts_z = V_dg_z_parent.element.interpolation_points
    
    expr_t = fem.Expression(E_opt.sub(0), pts_t)
    E_t_parent = fem.Function(V_dg_t_parent)
    E_t_parent.interpolate(expr_t)
    
    expr_z = fem.Expression(E_opt.sub(1), pts_z)
    E_z_parent = fem.Function(V_dg_z_parent)
    E_z_parent.interpolate(expr_z)

    V_sub_t = fem.functionspace(submesh, ("DG", 1, (2,)))
    V_sub_z = fem.functionspace(submesh, ("DG", 1))
    E_t_sub = fem.Function(V_sub_t)
    E_t_sub.interpolate(E_t_parent)
    E_z_sub = fem.Function(V_sub_z)
    E_z_sub.interpolate(E_z_parent)

    gamma_val = (2.0 * np.pi / lam) * n_eff
    
    # Обратное ВРМБ (BSBS)
    q_b = 0  
    print(f"Поиск механики (q_b = {q_b:.4f} мкм^-1)...")
    
    elastic_solver = ElasticSolver(submesh, mat_si.rho, C_si, q_b)
    # Судя по вашим логам, продольная мода BSBS в кремнии лежит около 24.5 ГГц
    mech_modes = elastic_solver.solve(guess_freq=24.5, n_modes=20)

    print("\n=== РАСЧЕТ УСИЛЕНИЯ ===")
    omega_opt = 2.0 * np.pi * const.c / (lam * 1e-6)
    gain_calc = BrillouinGainCalculator(submesh)
    mech_modes.sort(key=lambda x: x[1])
    
    for i, (U_mech, freq_mech) in enumerate(mech_modes):
        if freq_mech < 1.0: continue
            
        omega_mech = freq_mech * 2.0 * np.pi * 1e9
        power_mech = elastic_solver.calculate_power(U_mech, freq_mech)
        if power_mech == 0.0: continue
            
        # Замените 4.0 на 2.0 в gain_calc.py, если вы этого еще не сделали!
        # prefactor = (Q_mech * omega_opt) / (2.0 * (power_opt**2) * P_mech)
        gain_rp, gain_es, gain_total, ov_es, ov_rp = gain_calc.calculate_total_gain(
            E_t_sub, E_z_sub, U_mech, omega_opt, omega_mech, 
            Q_mech=249, power_opt=power_opt, P_mech=power_mech,
            mat_core=mat_si, mat_clad=mat_air, gamma_val=gamma_val, q_b=q_b
        )
        
        # Печатаем ВСЕ моды, чтобы видеть физическую картину
        print(f"\nМода {i+1}: {freq_mech:.4f} ГГц")
        print(f"  P_mech = {power_mech:.2e} J/m | Ov_ES = {ov_es:.2e} N | Ov_RP = {ov_rp:.2e} N")
        
        if gain_total > 1.0:
            print(f"  <--- НАЙДЕНА АКТИВНАЯ МОДА! Gain = {gain_total:6.1f} W^-1 m^-1 --->")
        else:
            print(f"  Gain = {gain_total:.2e} W^-1 m^-1")

    print(f"\n=== Расчет окончен! ===")

if __name__ == "__main__":
    main()


