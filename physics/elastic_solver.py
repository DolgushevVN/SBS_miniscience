import ufl
import basix.ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

def eps_T(u):
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), 0.0, u[2].dx(1), u[2].dx(0), u[0].dx(1) + u[1].dx(0)])

def eps_Z(u):
    return ufl.as_vector([0.0, 0.0, u[2], u[1], u[0], 0.0])

class ElasticSolver:
    def __init__(self, domain, cell_tags, mat_dict, q_b):
        self.domain = domain
        self.cell_tags = cell_tags
        self.mat_dict = mat_dict
        self.q_b = q_b
        
        v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(3,))
        self.V = fem.functionspace(domain, v_el)
        
        V_rho = fem.functionspace(domain, ("DG", 0))
        self.rho = fem.Function(V_rho)
        for tag, mat in mat_dict.items():
            cells = cell_tags.find(tag)
            if len(cells) > 0:
                self.rho.x.array[cells] = mat.rho
            
    def solve(self, guess_freq, n_modes=5):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        
        grad_u = eps_T(u) + 1j * self.q_b * eps_Z(u)
        grad_v = eps_T(v) + 1j * self.q_b * eps_Z(v)
        
        a = None
        for tag, mat in self.mat_dict.items():
            C_np = mat.get_stiffness_tensor()
            if mat.name == "Air": C_np = np.eye(6) * 1e-6
            
            C_mat = ufl.as_matrix(C_np.tolist())
            stress_u = ufl.dot(C_mat, grad_u)
            term = ufl.inner(stress_u, grad_v) * dx(tag)
            a = term if a is None else a + term
            
        b = self.rho * ufl.inner(u, v) * dx
        
        A = assemble_matrix(fem.form(a))
        A.assemble()
        B = assemble_matrix(fem.form(b))
        B.assemble()
        
        eigensolver = SLEPc.EPS().create()
        eigensolver.setOperators(A, B)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eigensolver.setDimensions(nev=n_modes)
        
        shift = (2.0 * np.pi * guess_freq)**2
        eigensolver.setTarget(shift)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        eigensolver.solve()
        
        if eigensolver.getConverged() > 0:
            val = eigensolver.getEigenvalue(0)
            freq = np.sqrt(max(0, val.real)) / (2 * np.pi)
            vr, vi = A.getVecs()
            eigensolver.getEigenvector(0, vr, vi)
            u_func = fem.Function(self.V)
            u_func.x.array[:] = vr.array + 1j * vi.array
            return u_func, freq
        else:
            raise RuntimeError("Упругий решатель не сошелся!")



