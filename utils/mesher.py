import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def create_waveguide_mesh(w_sim, h_sim, w_wg, h_wg, res_wg, res_clad):
    """
    Создает прямоугольный волновод внутри оболочки с помощью встроенных средств DOLFINx.
    Никаких внешних зависимостей типа gmsh!
    """
    # Определяем количество ячеек (задаем единое разрешение для простоты)
    nx = int(w_sim / res_wg)
    ny = int(h_sim / res_wg)
    
    # 1. Создаем базовую прямоугольную сетку
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-w_sim/2, -h_sim/2]), np.array([w_sim/2, h_sim/2])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )
    
    tdim = domain.topology.dim
    
    # --- 2. Размечаем подобласти (ядро и оболочка) ---
    def core_locator(x):
        # x[0] - X, x[1] - Y. Возвращает True для точек внутри волновода
        return (np.abs(x[0]) <= w_wg/2 + 1e-10) & (np.abs(x[1]) <= h_wg/2 + 1e-10)
    
    # Находим индексы всех ячеек (треугольников)
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    
    # Находим треугольники, которые лежат внутри ядра
    core_cells = mesh.locate_entities(domain, tdim, core_locator)
    
    # Создаем массив тегов: по умолчанию 1 (оболочка), для ядра ставим 2
    cell_tags_array = np.full(num_cells, 1, dtype=np.int32)
    cell_tags_array[core_cells] = 2
    
    # Запаковываем в объект FEniCSx
    cell_tags = mesh.meshtags(domain, tdim, cell_indices, cell_tags_array)
    
    # --- 3. Размечаем внутренние границы (границы волновода) ---
    # Это нужно для dS-интеграла (прыжок напряжений) в расчете Gain
    domain.topology.create_connectivity(tdim - 1, tdim)
    
    def interface_locator(x):
        # Левая и правая грани волновода
        on_lr = (np.abs(np.abs(x[0]) - w_wg/2) < 1e-5) & (np.abs(x[1]) <= h_wg/2 + 1e-5)
        # Верхняя и нижняя грани волновода
        on_tb = (np.abs(np.abs(x[1]) - h_wg/2) < 1e-5) & (np.abs(x[0]) <= w_wg/2 + 1e-5)
        return on_lr | on_tb
        
    interface_facets = mesh.locate_entities(domain, tdim - 1, interface_locator)
    
    # Помечаем интерфейс тегом 10
    facet_tags_array = np.full_like(interface_facets, 10, dtype=np.int32)
    facet_tags = mesh.meshtags(domain, tdim - 1, interface_facets, facet_tags_array)
    
    return domain, cell_tags, facet_tags


