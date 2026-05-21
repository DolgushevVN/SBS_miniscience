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
        Et, Ez = ufl.split(E_opt)
        E_vec = ufl.as_vector([Et[0], Et[1], Ez])
        E_sq = ufl.inner(E_vec, E_vec)
        
        p12 = mat_core.p12
        eps_core = mat_core.n**2
        
        # Объемная сила
        grad_E_sq = ufl.grad(E_sq)
        f_bulk = -0.5 * const.epsilon_0 * (eps_core**2) * p12 * ufl.as_vector([grad_E_sq[0], grad_E_sq[1], 0.0])
        overlap_bulk = fem.assemble_scalar(fem.form(ufl.inner(f_bulk, u_mech) * self.dx(2)))
        
        # Поверхностная сила (давление)
        n_2d = ufl.FacetNormal(self.domain)
        n_vec = ufl.as_vector([n_2d[0], n_2d[1], 0.0])
        
        V_eps = fem.functionspace(self.domain, ("DG", 0))
        eps_f = fem.Function(V_eps)
        eps_f.x.array[self.cell_tags.find(1)] = mat_clad.n**2
        eps_f.x.array[self.cell_tags.find(2)] = mat_core.n**2
        
        T = 0.5 * const.epsilon_0 * eps_f * (ufl.outer(E_vec, ufl.conj(E_vec)) - 0.5 * E_sq * ufl.Identity(3))
        overlap_bdr = fem.assemble_scalar(fem.form(ufl.inner(ufl.jump(T, n_vec), u_mech('-')) * self.dS(10)))
        
        total_overlap = overlap_bulk + overlap_bdr
        prefactor = (Q_mech * omega_opt) / (4.0 * (power_opt**2) * P_mech) * 1e21
        return prefactor * np.abs(total_overlap)**2


