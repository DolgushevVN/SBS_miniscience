import ufl
from dolfinx import fem
import numpy as np
import scipy.constants as const

class BrillouinGainCalculator:
    def __init__(self, domain, cell_tags, facet_tags):
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
        self.dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)

    def calculate_total_gain(self, E_opt, u_mech, omega_opt, omega_mech, Q_mech, power_opt, P_mech, mat_core, mat_clad):
        """
        Вычисляет полное усиление ВРМБ (Gain) в единицах W^-1 m^-1.
        """
        # Разделяем смешанное поле на поперечные и продольные компоненты
        Et, Ez = ufl.split(E_opt)
        
        # 1. Объемная сила электрострикции (Bulk Electrostriction)
        E_vec = ufl.as_vector([Et[0], Et[1], 1j * Ez])
        E_sq = ufl.inner(E_vec, E_vec)
        
        p12 = mat_core.p12
        eps_core = mat_core.n**2
        eps_clad = mat_clad.n**2
        
        # Сила f = -0.5 * eps0 * n^4 * p12 * grad(|E|^2)
        grad_E_sq = ufl.grad(E_sq)
        f_bulk = -0.5 * const.epsilon_0 * (eps_core**2) * p12 * ufl.as_vector([grad_E_sq[0], grad_E_sq[1], 0.0])
        
        # Интеграл перекрытия объемных сил (dx(2) - ядро кремния)
        overlap_bulk = fem.assemble_scalar(fem.form(ufl.inner(f_bulk, u_mech) * self.dx(2)))
        
        # 2. Поверхностное давление (Radiation Pressure на интерфейсе dS(10))
        # На внутреннем интерфейсе dS мы ОБЯЗАНЫ ограничить все члены стороной '-' (ядро)
        n_2d = ufl.FacetNormal(self.domain)
        n_vec = ufl.as_vector([n_2d[0], n_2d[1], 0.0])
        
        E_vec_bdr = ufl.as_vector([Et[0]('-'), Et[1]('-'), 1j * Ez('-')])
        E_sq_bdr = ufl.inner(E_vec_bdr, E_vec_bdr)
        
        # Вычисляем поверхностное натяжение (скачок тензора Максвелла)
        # n_vec('-') - это внешняя нормаль к кремниевому ядру
        overlap_bdr = fem.assemble_scalar(fem.form(
            0.5 * const.epsilon_0 * (eps_core - eps_clad) * E_sq_bdr * ufl.inner(n_vec('-'), u_mech('-')) * self.dS(10)
        ))
        
        # Суммируем вклады
        total_overlap = overlap_bulk + overlap_bdr
        
        # Коэффициент нормировки (перевод из мкм в метры через 1e21)
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt**2) * P_mech) * 1e21
        
        return prefactor * np.abs(total_overlap)**2


