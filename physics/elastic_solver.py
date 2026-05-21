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
    def __init__(self, domain, rho_val, stiffness_tensor, q_b):
        self.domain = domain
        self.q_b = q_b
        self.rho_val = rho_val
        self.C_np = stiffness_tensor
        
        v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(3,))
        self.V = fem.functionspace(domain, v_el)
            
    def solve(self, guess_freq, n_modes=1):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.domain)
        
        grad_u = eps_T(u) + 1j * self.q_b * eps_Z(u)
        grad_v = eps_T(v) + 1j * self.q_b * eps_Z(v)
        
        C_mat = ufl.as_matrix(self.C_np.tolist())
        stress_u = ufl.dot(C_mat, grad_u)
        
        a = ufl.inner(stress_u, grad_v) * dx
        b = self.rho_val * ufl.inner(u, v) * dx
        
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

    def calculate_power(self, u_func, freq_mech):
        omega = 2.0 * np.pi * freq_mech * 1e9
        dx = ufl.Measure("dx", domain=self.domain)
        term = 0.5 * (omega**2) * self.rho_val * ufl.inner(u_func, u_func) * dx
        return fem.assemble_scalar(fem.form(term)).real


