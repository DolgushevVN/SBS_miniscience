import ufl
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix
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
        self.k0_sq = (2.0 * np.pi / wavelength)**2
        
        # Правильное создание смешанного пространства элементов
        cell_name = domain.topology.cell_name()
        N = basix.ufl.element("N1curl", cell_name, 1)
        L = basix.ufl.element("Lagrange", cell_name, 1)
        mix_el = basix.ufl.mixed_element([N, L])
        self.V = fem.functionspace(domain, mix_el)
        
        V_eps = fem.functionspace(domain, ("DG", 0))
        self.eps_r = fem.Function(V_eps)
        for tag, mat in mat_dict.items():
            cells = cell_tags.find(tag)
            if len(cells) > 0:
                self.eps_r.x.array[cells] = mat.n**2

    def solve(self, guess_neff, n_modes=2):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        Et, Ez = ufl.split(u)
        Vt, Vz = ufl.split(v)
        
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        u_r = 1.0 
        
        s_tt = (1.0/u_r) * ufl.inner(curl_t(Et), curl_t(Vt))
        t_tt = self.eps_r * ufl.inner(Et, Vt)
        s_zz = (1.0/u_r) * ufl.inner(ufl.grad(Ez), ufl.grad(Vz))
        t_zz = self.eps_r * ufl.inner(Ez, Vz) 
        
        b_tt = (1.0/u_r) * ufl.inner(Et, Vt)
        b_tz = (1.0/u_r) * ufl.inner(Et, ufl.grad(Vz))
        b_zt = (1.0/u_r) * ufl.inner(ufl.grad(Ez), Vt)
        
        a = (s_tt - self.k0_sq * t_tt) * dx
        b = (s_zz - self.k0_sq * t_zz + b_tt + b_tz + b_zt) * dx
        
        # Граничные условия
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)
        boundary_facets = dmesh.exterior_facet_indices(self.domain.topology)
        
        bcs = []
        if len(boundary_facets) > 0:
            for i in range(2): # Для Et и для Ez
                V_sub, _ = self.V.sub(i).collapse()
                dofs = fem.locate_dofs_topological((self.V.sub(i), V_sub), self.domain.topology.dim - 1, boundary_facets)
                val = fem.Function(V_sub)
                val.x.array[:] = 0.0
                bcs.append(fem.dirichletbc(val, dofs, self.V.sub(i)))
        
        A = assemble_matrix(fem.form(a), bcs=bcs)
        A.assemble()
        
        B = assemble_matrix(fem.form(b))
        B.assemble() # СНАЧАЛА СОБИРАЕМ
        
        bc_indices = []
        for bc in bcs:
            dofs = bc.dof_indices()
            if isinstance(dofs, tuple): dofs = dofs[0]
            bc_indices.append(np.asarray(dofs, dtype=np.int32))
        
        if len(bc_indices) > 0:
            flattened_dofs = np.unique(np.concatenate(bc_indices)).astype(np.int32)
            B.zeroRowsLocal(flattened_dofs, diag=0.0) # ТЕПЕРЬ ОБНУЛЯЕМ
            B.assemble() # И ЕЩЕ РАЗ ФИКСИРУЕМ
        
        eigensolver = SLEPc.EPS().create()
        eigensolver.setOperators(A, B)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eigensolver.setDimensions(nev=n_modes)
        
        shift = (2.0 * np.pi * guess_neff / self.wavelength)**2
        eigensolver.setTarget(shift)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        eigensolver.solve()
        
        if eigensolver.getConverged() > 0:
            val = eigensolver.getEigenvalue(0)
            neff = np.sqrt(val.real) * self.wavelength / (2 * np.pi)
            vr, vi = A.getVecs()
            eigensolver.getEigenvector(0, vr, vi)
            E_func = fem.Function(self.V)
            E_func.x.array[:] = vr.array + 1j * vi.array
            return E_func, neff
        else:
            raise RuntimeError("ЭМ-решатель не сошелся!")


