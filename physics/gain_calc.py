import ufl
from dolfinx import fem
import numpy as np
import scipy.constants as const

class BrillouinGainCalculator:
    def __init__(self, submesh):
        self.submesh = submesh
        self.dx = ufl.Measure("dx", domain=submesh)
        self.ds = ufl.Measure("ds", domain=submesh) 

    def calculate_total_gain(self, E_t_sub, E_z_sub, u_mech, omega_opt, omega_mech, Q_mech, power_opt, P_mech, mat_core, mat_clad, gamma_val):
        # Восстанавливаем компоненты тензора |Ei|^2 с учетом фазового сдвига 1j * gamma для Ez
        EE = ufl.as_vector([
            ufl.real(E_t_sub[0] * ufl.conj(E_t_sub[0])),
            ufl.real(E_t_sub[1] * ufl.conj(E_t_sub[1])),
            ufl.real((gamma_val**2) * E_z_sub * ufl.conj(E_z_sub)),
            ufl.real(E_t_sub[0] * ufl.conj(E_t_sub[1]))
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
        
        # 1. Объемная электрострикция (Bulk ES)
        f_bulk = ufl.as_vector([
            s0.dx(0) + s5.dx(1),
            s5.dx(0) + s1.dx(1),
            0.0
        ])
        overlap_bulk = fem.assemble_scalar(fem.form(ufl.inner(f_bulk, u_mech) * self.dx))
        
        # Нормаль к границе волновода
        n_vec_2d = ufl.FacetNormal(self.submesh)
        n_vec = ufl.as_vector([n_vec_2d[0], n_vec_2d[1], 0.0])
        
        # 2. Граничная электрострикция (Boundary ES)
        f_bes = ufl.as_vector([
            -(s0 * n_vec[0] + s5 * n_vec[1]),
            -(s5 * n_vec[0] + s1 * n_vec[1]),
            0.0
        ])
        overlap_bes = fem.assemble_scalar(fem.form(ufl.inner(f_bes, u_mech) * self.ds))

        # 3. Радиационное давление (RP)
        En = E_t_sub[0]*n_vec[0] + E_t_sub[1]*n_vec[1]
        En_sq = ufl.real(En * ufl.conj(En))
        E_par_sq = E_sq - En_sq
        
        P_RP = 0.5 * const.epsilon_0 * (
            (eps_core - eps_clad) * E_par_sq + 
            (eps_core**2 / eps_clad - eps_core) * En_sq
        )
        overlap_rp = fem.assemble_scalar(fem.form(P_RP * ufl.inner(n_vec, u_mech) * self.ds))
        
        # === ПРЕОБРАЗОВАНИЕ В СТРОГИЕ СИ ===
        overlap_bulk_SI = overlap_bulk * 1e-6
        overlap_bes_SI = overlap_bes * 1e-6
        overlap_rp_SI = overlap_rp * 1e-6
        
        power_opt_SI = power_opt * 1e-12
        P_mech_SI = P_mech * 1e-9
        
        # Теоретический префактор (строго по уравнению 14 из статьи)
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt_SI**2) * P_mech_SI)
        
        gain_rp = prefactor * np.abs(overlap_rp_SI)**2
        gain_bulk = prefactor * np.abs(overlap_bulk_SI)**2
        gain_bes = prefactor * np.abs(overlap_bes_SI)**2
        gain_total = prefactor * np.abs(overlap_bulk_SI + overlap_rp_SI + overlap_bes_SI)**2
        
        return gain_rp, gain_bulk, gain_bes, gain_total

