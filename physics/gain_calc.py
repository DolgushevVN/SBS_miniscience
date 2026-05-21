import ufl
from dolfinx import fem
import numpy as np
import scipy.constants as const

class BrillouinGainCalculator:
    def __init__(self, submesh):
        self.submesh = submesh
        self.dx = ufl.Measure("dx", domain=submesh)
        self.ds = ufl.Measure("ds", domain=submesh) 

    def calculate_total_gain(self, E_opt_sub, u_mech, omega_opt, omega_mech, Q_mech, power_opt, P_mech, mat_core, mat_clad):
        # Вычисляем квадраты модулей полей |Ei|^2
        EE = ufl.as_vector([
            ufl.real(E_opt_sub[0] * ufl.conj(E_opt_sub[0])),
            ufl.real(E_opt_sub[1] * ufl.conj(E_opt_sub[1])),
            ufl.real(E_opt_sub[2] * ufl.conj(E_opt_sub[2])),
            ufl.real(E_opt_sub[0] * ufl.conj(E_opt_sub[1]))
        ])
        
        E_sq = EE[0] + EE[1] + EE[2]
        
        p11 = mat_core.p11
        p12 = mat_core.p12
        p44 = -0.051
        eps_core = mat_core.n**2
        eps_clad = mat_clad.n**2
        
        # Напряжения электрострикции
        s0 = -0.5 * const.epsilon_0 * (eps_core**2) * (p11 * EE[0] + p12 * EE[1] + p12 * EE[2])
        s1 = -0.5 * const.epsilon_0 * (eps_core**2) * (p12 * EE[0] + p11 * EE[1] + p12 * EE[2])
        s5 = -0.5 * const.epsilon_0 * (eps_core**2) * (p44 * 2.0 * EE[3])
        
        # Объемная сила (Честная физика, без коэффициента 4.43)
        f_bulk = ufl.as_vector([
            s0.dx(0) + s5.dx(1),
            s5.dx(0) + s1.dx(1),
            0.0
        ])
        overlap_bulk = fem.assemble_scalar(fem.form(ufl.inner(f_bulk, u_mech) * self.dx))
        
        # Стабильная модель радиационного давления на границе ds
        n_vec_2d = ufl.FacetNormal(self.submesh)
        n_vec = ufl.as_vector([n_vec_2d[0], n_vec_2d[1], 0.0])
        
        overlap_bdr = fem.assemble_scalar(fem.form(
            0.5 * const.epsilon_0 * (eps_core - eps_clad) * E_sq * ufl.inner(n_vec, u_mech) * self.ds
        ))
        
        # Отладочный вывод
        print(f"[DEBUG] Поток мощности P_opt: {power_opt:.5e} Вт")
        print(f"[DEBUG] Механическая энергия P_mech: {P_mech:.5e} Дж")
        print(f"[DEBUG] Сырой интеграл Bulk Overlap: {overlap_bulk:.5e}")
        print(f"[DEBUG] Сырой интеграл Boundary Overlap: {overlap_bdr:.5e}")

        # Теоретический префактор (строго по уравнению 14 из статьи)
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt**2) * P_mech)
        
        gain_bdr = prefactor * np.abs(overlap_bdr)**2
        gain_bulk = prefactor * np.abs(overlap_bulk)**2
        gain_total = prefactor * np.abs(overlap_bulk + overlap_bdr)**2
        
        return gain_bdr, gain_bulk, gain_total


