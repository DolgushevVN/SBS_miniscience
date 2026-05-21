import numpy as np

class Material:
    def __init__(self, name, rho, M, G, n_ref, p11, p12):
        self.name = name
        self.rho = rho      # Плотность
        self.M = M          # Продольный модуль (P-wave)
        self.G = G          # Модуль сдвига (Shear)
        self.n = n_ref      # Показатель преломления
        self.p11 = p11      # Фотоупругость
        self.p12 = p12
        
    def get_stiffness_tensor(self):
        # Собираем тензор 6x6 по Фойгту
        M, G = self.M, self.G
        return np.array([
            [M, M-2*G, M-2*G, 0, 0, 0],
            [M-2*G, M, M-2*G, 0, 0, 0],
            [M-2*G, M-2*G, M, 0, 0, 0],
            [0, 0, 0, G, 0, 0],
            [0, 0, 0, 0, G, 0],
            [0, 0, 0, 0, 0, G]
        ])

