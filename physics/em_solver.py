import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

def curl_t(w):
    """Поперечный ротор 2D вектора"""
    return w[1].dx(0) - w[0].dx(1)

class EMSolver:
    def __init__(self, domain, cell_tags, mat_dict, wavelength):
        self.domain = domain
        self.cell_tags = cell_tags
        self.wavelength = wavelength
        self.k0_sq = (2.0 * np.pi / wavelength)**2
        
        # Смешанное пространство: Неделек (поперечное поле Et) + Лагранж (продольное Ez)
        # В FEniCSx смешанные элементы задаются так:
        N = ufl.FiniteElement("Nedelec 1st kind H(curl)", domain.ufl_cell(), 1)
        L = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        mix_el = ufl.MixedElement([N, L])
        self.V = fem.functionspace(domain, mix_el)
        
        # Задаем диэлектрическую проницаемость eps_r (кусочно-постоянную)
        V_eps = fem.functionspace(domain, ("DG", 0))
        self.eps_r = fem.Function(V_eps)
        for tag, mat in mat_dict.items():
            cells = cell_tags.find(tag)
            self.eps_r.x.array[cells] = mat.n**2  # eps = n^2

    def solve(self, guess_neff, n_modes=2):
        # Пробные и тестовые функции
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Разделяем на поперечную (t) и продольную (z) части
        Et, Ez = ufl.split(u)
        Vt, Vz = ufl.split(v)
        
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        
        # Слабая форма электромагнитного уравнения (из диссертации Elan Lezar)
        # Обрати внимание: в FEniCSx мы пишем математику напрямую!
        u_r = 1.0 # магнитная проницаемость
        
        s_tt = (1.0/u_r) * ufl.inner(curl_t(Et), curl_t(Vt))
        t_tt = self.eps_r * ufl.inner(Et, Vt)
        s_zz = (1.0/u_r) * ufl.inner(ufl.grad(Ez), ufl.grad(Vz))
        t_zz = self.eps_r * Ez * Vz
        
        b_tt = (1.0/u_r) * ufl.inner(Et, Vt)
        b_tz = (1.0/u_r) * ufl.inner(Et, ufl.grad(Vz))
        b_zt = (1.0/u_r) * ufl.inner(ufl.grad(Ez), Vt)
        
        # Уравнения для левой (A) и правой (B) матриц
        a = (s_tt - self.k0_sq * t_tt) * dx
        b = (s_zz - self.k0_sq * t_zz + b_tt + b_tz + b_zt) * dx
        
        # Сборка матриц PETSc
        A = assemble_matrix(fem.form(a))
        A.assemble()
        B = assemble_matrix(fem.form(b))
        B.assemble()
        
        # Настройка SLEPc решателя (поиск собственных значений)
        eigensolver = SLEPc.EPS().create()
        eigensolver.setOperators(A, B)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eigensolver.setDimensions(nev=n_modes)
        
        # Ищем моду вокруг эффективного показателя преломления (guess_neff)
        shift = -(2.0 * np.pi * guess_neff / self.wavelength)**2
        eigensolver.setTarget(shift)
        eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        
        st = eigensolver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        eigensolver.solve()
        
        # Возвращаем оптическое поле E и эффективный индекс
        if eigensolver.getConverged() > 0:
            val = eigensolver.getEigenvalue(0)
            neff = np.sqrt(-val.real) * self.wavelength / (2 * np.pi)
            
            # Извлекаем вектор поля
            vr, vi = A.getVecs()
            eigensolver.getEigenvector(0, vr, vi)
            
            E_func = fem.Function(self.V)
            E_func.x.array[:] = vr.array + 1j * vi.array
            return E_func, neff
        else:
            raise RuntimeError("ЭМ-решатель не сошелся!")

