# Структура файлов

- `papers/` — тут лежат статьи
- `report/` — тут лежит все для финального отчета

# Как запустить 

```
# Создаем новое окружение (Python 3.10-3.12)
conda create -n fenicsx-complex python=3.11
conda activate fenicsx-complex

# Устанавливаем DOLFINx с поддержкой комплексных чисел
conda install -c conda-forge fenics-dolfinx petsc=*=*complex* slepc=*=*complex* petsc4py slepc4py

# Устанавливаем Gmsh для генерации сеток и остальные нужные библиотеки
conda install -c conda-forge gmsh python-gmsh scipy matplotlib numpy
```
