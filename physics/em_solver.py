import ufl
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix
import scipy.constants as const
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

def curl_t(w):
    return w[1].dx(0) - w[0].dx(1)

class EMSolver:
    def __init__(self, domain, cell_tags, mat_dict, wavelength):
        self.domain = domain
        self.cell_tags = cell_tags
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.k0_sq = self.k0**2
        
        cell_name = domain.topology.cell_name()
        N = basix.ufl.element("N1curl", cell_name, 1)
        L = basix.ufl.element("Lagrange", cell_name, 1)
        self.V = fem.functionspace(domain, basix.ufl.mixed_element([N, L]))
        
        V_eps = fem.functionspace(domain, ("DG", 0))
        self.eps_r = fem.Function(V_eps)
        for tag, mat in mat_dict.items():
            cells = cell_tags.find(tag)
            if len(cells) > 0:
                self.eps_r.x.array[cells] = mat.n**2

    def solve(self, guess_neff, n_modes=1):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        Et, Ez = ufl.split(u)
        Vt, Vz = ufl.split(v)
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        
        s_tt = ufl.inner(curl_t(Et), curl_t(Vt))
        t_tt = self.eps_r * ufl.inner(Et, Vt)
        s_zz = ufl.inner(ufl.grad(Ez), ufl.grad(Vz))
        t_zz = self.eps_r * ufl.inner(Ez, Vz) 
        b_tt = ufl.inner(Et, Vt)
        b_tz = ufl.inner(Et, ufl.grad(Vz))
        b_zt = ufl.inner(ufl.grad(Ez), Vt)
        
        a = (s_tt - self.k0_sq * t_tt) * dx + ufl.inner(1e-10 * Ez, Vz) * dx
        b = (s_zz - self.k0_sq * t_zz + b_tt + b_tz + b_zt) * dx
        
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)
        boundary_facets = dmesh.exterior_facet_indices(self.domain.topology)
        
        bcs = []
        if len(boundary_facets) > 0:
            for i in range(2):
                V_sub, _ = self.V.sub(i).collapse()
                dofs = fem.locate_dofs_topological((self.V.sub(i), V_sub), self.domain.topology.dim - 1, boundary_facets)
                val = fem.Function(V_sub)
                bcs.append(fem.dirichletbc(val, dofs, self.V.sub(i)))
        
        A = assemble_matrix(fem.form(a), bcs=bcs)
        A.assemble()
        B = assemble_matrix(fem.form(b), bcs=bcs)
        
        bc_dofs = []
        for bc in bcs:
            bc_dofs.extend(bc.dof_indices()[0] if isinstance(bc.dof_indices(), tuple) else bc.dof_indices())
        B.assemble()
        B.zeroRowsLocal(np.array(bc_dofs, dtype=np.int32), diag=0.0)
        B.assemble()
        
        eigensolver = SLEPc.EPS().create()
        eigensolver.setOperators(A, B)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        
        target = -((2.0 * np.pi * guess_neff / self.wavelength)**2)
        eigensolver.setTarget(target)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        eigensolver.solve()
        
        if eigensolver.getConverged() > 0:
            eig_val = eigensolver.getEigenvalue(0)
            gamma = np.sqrt(abs(eig_val.real))
            neff = gamma / self.k0
            
            vr, vi = A.getVecs()
            eigensolver.getEigenvector(0, vr, vi)
            E_func = fem.Function(self.V)
            E_func.x.array[:] = vr.array + 1j * vi.array
            return E_func, neff
        else:
            raise RuntimeError("ЭМ-решатель не сошелся!")

    def calculate_power(self, E_func, neff, ng):
        """Вычисляет оптическую мощность ненормированной моды"""
        Et, Ez = ufl.split(E_func)
        gamma_sq = (self.k0 * neff)**2
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        
        power = 0.0
        for tag in [1, 2]: # Интегрируем по воздуху (1) и кремнию (2)
            cells = self.cell_tags.find(tag)
            if len(cells) > 0:
                eps_val = 1.0 if tag == 1 else 3.48**2
                term = 0.5 * (const.c / ng) * const.epsilon_0 * eps_val * (ufl.inner(Et, Et) + gamma_sq * ufl.inner(Ez, Ez)) * dx(tag)
                power += fem.assemble_scalar(fem.form(term)).real
        return power


