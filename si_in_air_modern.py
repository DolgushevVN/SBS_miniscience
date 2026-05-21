import numpy as np
import scipy.constants as const

# Импортируем наши новые модули (предполагаем, что они лежат в папках)
from utils.mesher import create_waveguide_mesh
from utils.materials import Material
from physics.em_solver import EMSolver
from physics.elastic_solver import ElasticSolver
from physics.gain_calc import BrillouinGainCalculator

def main():
    print("=== Начало расчета ВРМБ (Кремний в воздухе) ===")
    
    # 1. Параметры геометрии (мкм)
    w_sim, h_sim = 1.5, 1.5
    w_wg, h_wg = 0.45, 0.315
    res_wg = 0.02   # Высокое разрешение внутри кремния
    res_clad = 0.1  # Низкое разрешение в воздухе

    # 2. Создаем материалы
    # Silicon (Кремний) - тег 2
    mat_si = Material(
        name="Silicon", rho=2.328, M=165.7, G=79.6, 
        n_ref=3.48, p11=-0.09, p12=0.017
    )
    # Air (Воздух) - тег 1
    mat_air = Material(
        name="Air", rho=1.2e-3, M=0.0, G=0.0, 
        n_ref=1.0, p11=0.0, p12=0.0
    )
    materials = {1: mat_air, 2: mat_si}

    # 3. Генерируем сетку через Gmsh
    print("Генерация сетки...")
    domain, cell_tags, facet_tags = create_waveguide_mesh(w_sim, h_sim, w_wg, h_wg, res_wg, res_clad)

    # 4. Оптическая часть (Свет)
    wavelength = 1.55 # мкм
    print(f"Поиск оптической моды (длина волны {wavelength} мкм)...")
    em_solver = EMSolver(domain, cell_tags, materials, wavelength)
    
    # Решаем с догадкой neff = n_si^2 (3.48^2 = ~12.1)
    E_opt, n_eff = em_solver.solve(guess_neff=12.1, n_modes=1)
    print(f"Оптическая мода найдена! Эффективный индекс n_eff = {n_eff:.4f}")

    # 5. Акустическая часть (Звук)
    # Звуковая волна (forward scattering: q_b ~ 0)
    q_b = 0.0 
    print("Поиск упругой (акустической) моды...")
    elastic_solver = ElasticSolver(domain, cell_tags, materials, q_b)
    
    # В старом коде догадка была 8.5 ГГц
    U_mech, freq_mech = elastic_solver.solve(guess_freq=8.5, n_modes=1)
    print(f"Упругая мода найдена! Частота = {freq_mech:.4f} ГГц")

    # 6. Расчет усиления (Gain)
    print("Вычисление усиления ВРМБ (Gain)...")
    omega_opt = 2.0 * np.pi * const.c / (wavelength * 1e-6)
    omega_mech = freq_mech * 2.0 * np.pi * 1e9
    
    gain_calc = BrillouinGainCalculator(domain, cell_tags, facet_tags)
    
    # Для теста ставим условные значения мощностей P=1W
    gain = gain_calc.calculate_total_gain(
        E_opt, U_mech, omega_opt, omega_mech, 
        Q_mech=1000, power_opt=1.0, P_mech=1.0, 
        mat_core=mat_si, mat_clad=mat_air
    )
    
    print(f"=== Расчет окончен! ===")
    print(f"Итоговое усиление: {gain:.2f} W^-1 m^-1")

if __name__ == "__main__":
    main()


