import ufl
from dolfinx import fem
import numpy as np
import scipy.constants as const

def grad_3d(u, q_b):
    """Строит полный 3D градиент поля (учитывая фазу e^{iqz} вдоль волновода)"""
    return ufl.as_tensor([
        [u[0].dx(0), u[0].dx(1), 1j * q_b * u[0]],
        [u[1].dx(0), u[1].dx(1), 1j * q_b * u[1]],
        [u[2].dx(0), u[2].dx(1), 1j * q_b * u[2]]
    ])

class BrillouinGainCalculator:
    def __init__(self, submesh):
        self.submesh = submesh
        self.dx = ufl.Measure("dx", domain=submesh)
        self.ds = ufl.Measure("ds", domain=submesh) 

    def calculate_total_gain(self, E_t_sub, E_z_sub, u_mech, omega_opt, omega_mech, Q_mech, power_opt, P_mech, mat_core, mat_clad, gamma_val, q_b):
        
        # 1. Сборка истинного 3D оптического вектора
        Ex = E_t_sub[0]
        Ey = E_t_sub[1]
        Ez_phys = 1j * gamma_val * E_z_sub
        E_vec = ufl.as_vector([Ex, Ey, Ez_phys])
        
        # 2. Компоненты оптической интенсивности EE_ij = E_i * E_j^*
        EE0 = Ex * ufl.conj(Ex)         # |Ex|^2
        EE1 = Ey * ufl.conj(Ey)         # |Ey|^2
        EE2 = Ez_phys * ufl.conj(Ez_phys) # |Ez|^2
        EE3 = Ex * ufl.conj(Ey)         # Ex Ey*
        EE4 = Ex * ufl.conj(Ez_phys)    # Ex Ez*
        EE5 = Ey * ufl.conj(Ez_phys)    # Ey Ez*
        
        p11, p12, p44 = mat_core.p11, mat_core.p12, -0.051
        eps_core, eps_clad = mat_core.n**2, mat_clad.n**2
        pre = -0.5 * const.epsilon_0 * (eps_core**2)
        
        # 3. ПОЛНЫЙ 3x3 Тензор напряжений Электрострикции (T_ij)
        Txx = pre * (p11*EE0 + p12*EE1 + p12*EE2)
        Tyy = pre * (p12*EE0 + p11*EE1 + p12*EE2)
        Tzz = pre * (p12*EE0 + p12*EE1 + p11*EE2) # <-- Раньше мы это полностью игнорировали!
        Txy = pre * p44 * 2.0 * EE3
        Txz = pre * p44 * 2.0 * EE4               # <-- И это!
        Tyz = pre * p44 * 2.0 * EE5               # <-- И это!
        
        Tyx = ufl.conj(Txy)
        Tzx = ufl.conj(Txz)
        Tzy = ufl.conj(Tyz)
        
        T_tensor = ufl.as_tensor([
            [Txx, Txy, Txz],
            [Tyx, Tyy, Tyz],
            [Tzx, Tzy, Tzz]
        ])
        
        # 4. Вычисление работы электрострикции через скалярное произведение 3x3 тензоров
        # inner(A, B) делает A : B^* автоматически
        W_es = -ufl.inner(T_tensor, grad_3d(u_mech, q_b))
        overlap_es_raw = fem.assemble_scalar(fem.form(W_es * self.dx)) * 1e-6
        
        # 5. Радиационное давление (RP)
        n_vec_2d = ufl.FacetNormal(self.submesh)
        n_vec = ufl.as_vector([n_vec_2d[0], n_vec_2d[1], 0.0])
        
        E_sq = ufl.inner(E_vec, E_vec) 
        En = ufl.inner(E_vec, n_vec)
        En_sq = ufl.inner(En, En)
        E_par_sq = E_sq - En_sq
        
        P_RP = 0.5 * const.epsilon_0 * (
            (eps_core - eps_clad) * E_par_sq + 
            (eps_core**2 / eps_clad - eps_core) * En_sq
        )
        overlap_rp_raw = fem.assemble_scalar(fem.form(P_RP * ufl.inner(n_vec, u_mech) * self.ds)) * 1e-6
        
        # 6. Сборка итогового Gain
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt**2) * P_mech)
        
        gain_rp = prefactor * np.abs(overlap_rp_raw)**2
        gain_es = prefactor * np.abs(overlap_es_raw)**2
        gain_total = prefactor * np.abs(overlap_es_raw + overlap_rp_raw)**2
        
        return gain_rp, gain_es, gain_total, np.abs(overlap_es_raw), np.abs(overlap_rp_raw)


