import ufl
from dolfinx import fem
import numpy as np
import scipy.constants as const

class BrillouinGainCalculator:
    def __init__(self, domain, cell_tags, facet_tags):
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags # Понадобится для интегралов по границам (dS)
        self.dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
        self.dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)

    def calculate_total_gain(self, E_opt, u_mech, omega_opt, omega_mech, Q_mech, power_opt, P_mech, mat_core, mat_clad):
        """
        Полный расчет усиления ВРМБ, объединяющий объемную электрострикцию
        и радиационное давление на границах.
        """
        # 1. Извлекаем компоненты полей
        Et, Ez = ufl.split(E_opt)
        E_vec = ufl.as_vector([Et[0], Et[1], Ez])
        
        # 2. Объемная электрострикция (Bulk Electrostriction)
        # В UFL мы можем написать тензор прямо как математическую матрицу:
        p11, p12 = mat_core.p11, mat_core.p12
        eps_core = mat_core.n**2
        
        # Для простоты показываем скалярный пример сжатия поля.
        # В полной версии здесь перемножение тензора p_ijkl на E_k * E_l^*
        E_sq = ufl.inner(E_vec, E_vec)
        # Упрощенная сила электрострикции (градиент плотности энергии):
        f_bulk = -0.5 * const.epsilon_0 * (eps_core**2) * p12 * ufl.grad(E_sq)
        
        # Интеграл перекрытия объема: \int (f_bulk * u_mech^*) dx
        overlap_bulk = fem.assemble_scalar(fem.form(ufl.inner(f_bulk, u_mech) * self.dx(2)))
        
        # 3. Радиационное давление на границе (Radiation Pressure)
        # Прыжок тензора напряжений Максвелла на интерфейсе ядро(2)-оболочка(1)
        # n_vec - вектор нормали к границе
        n_vec = ufl.FacetNormal(self.domain)
        
        # Вычисляем тензор напряжений T_ij для кремния (+) и воздуха (-)
        T_core = 0.5 * const.epsilon_0 * eps_core * (ufl.outer(E_vec, E_vec) - 0.5 * E_sq * ufl.Identity(3))
        T_clad = 0.5 * const.epsilon_0 * (mat_clad.n**2) * (ufl.outer(E_vec, E_vec) - 0.5 * E_sq * ufl.Identity(3))
        
        # Прыжок напряжений на внутренней границе (dS)
        f_boundary = ufl.jump(T_core, n_vec) - ufl.jump(T_clad, n_vec)
        
        # Интеграл перекрытия поверхности (Boundary coupling): \int (f_boundary * u_mech^*) dS
        # В реальной задаче граница (интерфейс) должна быть помечена тегом в facet_tags (например, тег 10)
        overlap_bdr = fem.assemble_scalar(fem.form(ufl.inner(f_boundary, u_mech('-')) * self.dS(10)))
        
        # 4. Итоговое усиление (формула 14 из статьи)
        total_overlap = overlap_bulk + overlap_bdr
        
        # Префактор (учитывающий качество резонанса Q_mech и мощности)
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt**2) * P_mech) * 1e21
        
        gain = prefactor * np.abs(total_overlap)**2
        return gain


