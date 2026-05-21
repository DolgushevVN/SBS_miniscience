import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix # Прямой импорт!
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
        self.q_b = q_b
        
        # Надежный способ создания векторного пространства для новых версий
        v_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
        self.V = fem.functionspace(domain, v_el)
        
        self.rho = fem.Function(fem.functionspace(domain, ("DG", 0)))
        
        for tag, mat in mat_dict.items():
            cells = cell_tags.find(tag)
            self.rho.x.array[cells] = mat.rho
            
    def solve(self, guess_freq, n_modes=5):
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        
        C_dummy = 1.0 
        
        grad_u = eps_T(u) + 1j * self.q_b * eps_Z(u)
        grad_v = eps_T(v) + 1j * self.q_b * eps_Z(v)
        
        a = ufl.inner(C_dummy * grad_u, grad_v) * dx
        b = self.rho * ufl.inner(u, v) * dx
        
        # Используем assemble_matrix напрямую
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
        
        print(f"Найдено акустических мод: {eigensolver.getConverged()}")
        if eigensolver.getConverged() > 0:
            val = eigensolver.getEigenvalue(0)
            freq = np.sqrt(val.real) / (2 * np.pi)
            
            # Достаем собственную функцию
            vr, vi = A.getVecs()
            eigensolver.getEigenvector(0, vr, vi)
            u_func = fem.Function(self.V)
            u_func.x.array[:] = vr.array + 1j * vi.array
            
            return u_func, freq
        else:
            raise RuntimeError("Упругий решатель не сошелся!")

