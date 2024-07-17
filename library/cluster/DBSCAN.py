"""
    1) Epsilon (ε): Этот параметр определяет радиус окрестности для точки, в пределах которого будут
проверяться соседние точки. Это расстояние, в пределах которого точки считаются соседями. По сути,
это максимальное расстояние между двумя точками, чтобы одна считалась в окрестности другой.

    2) MinPts (n): Этот параметр задает минимальное количество точек (включая саму точку), которые должны
находиться в ε-окрестности точки, чтобы она считалась "основной" точкой (core point). Если в пределах
ε-окрестности находится меньше, чем MinPts точек, то точка считается "пограничной" (border point) или
"шумом" (noise point).

"""

import numpy as np
from sklearn.cluster import DBSCAN

# Определение гиперпараметров:
# 1) eps - радиус окрестности для проверки;
# 2) min_samples - минимальное количество соседей для основной точки;
pairs_of_hyperparams = [
    [0.1, 20],
    [0.1, 50],
    [0.2, 20],
    [0.2, 50],
    [0.3, 20],
    [0.3, 50],
]

# Пример данных (замените на ваш набор данных)
dataset = ...  # Ваш набор данных
X = dataset[["x1", "x2"]]

# Список для хранения результатов кластеризации
results = []

for eps, min_samples in pairs_of_hyperparams:
    # Обучение модели DBSCAN с текущими гиперпараметрами
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    # Сохранение результатов кластеризации
    clusters = dbscan.labels_
    results.append({
        'eps': eps,
        'min_samples': min_samples,
        'clusters': clusters
    })

# Пример использования результатов:
for result in results:
    print(f"Epsilon: {result['eps']}, Min_samples: {result['min_samples']}, Number of clusters: {len(set(result['clusters'])) - (1 if -1 in result['clusters'] else 0)}")


#^ ===============>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def plot_dbscan_clusters(dataset, pairs_of_hyperparams, x_col='x1', y_col='x2', x1_min=None, x1_max=None, x2_min=None, x2_max=None):
    """
    Функция для визуализации кластеров, полученных методом DBSCAN, с различными парами гиперпараметров.

    Параметры:
    - dataset: DataFrame с данными, содержащий колонки для кластеризации.
    - pairs_of_hyperparams: Список пар [eps, min_samples] для DBSCAN.
    - x_col: Название колонки с признаками по оси X (по умолчанию 'x1').
    - y_col: Название колонки с признаками по оси Y (по умолчанию 'x2').
    - x1_min, x1_max: Минимальные и максимальные значения для оси X (необязательно).
    - x2_min, x2_max: Минимальные и максимальные значения для оси Y (необязательно).
    """
    
    fig = plt.figure()
    fig.set_size_inches(20, 50)

    for i, s in enumerate(pairs_of_hyperparams):
        
        X = dataset[[x_col, y_col]]
        
        dbscan = DBSCAN(eps=s[0], min_samples=s[1]).fit(X)
        ax_ = fig.add_subplot(5, 2, i + 1)
        
        colors = ["black", "#FF5533", "#00B050", "orange", "blue", "purple"]
        
        y = dbscan.labels_
        labels = sorted(list(set(dbscan.labels_)))
        
        if len(labels) == 1 and y[0] == 0:
            color_map = {0: "#FF5533"}
        else:
            color_map = dict(zip(labels, colors))
        
        for label, color in color_map.items():
            idx = np.where(y == label)
            plt.scatter(
                X.values[idx, 0],
                X.values[idx, 1],
                c=color,
                s=20,
                edgecolor="k",
                label=f'Cluster {label}' if label != -1 else 'Noise'
            )
        
        if x1_min is not None and x1_max is not None:
            plt.xlim(x1_min, x1_max)
        if x2_min is not None and x2_max is not None:
            plt.ylim(x2_min, x2_max)
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        n_clusters = len(set(y)) - (1 if -1 in y else 0)
        plt.title(f"Epsilon = {s[0]}\nmin_samples = {s[1]}\nwith {n_clusters} clusters")
    
    fig.tight_layout()
    plt.show()

# Пример использования функции без указания пределов осей
"""
pairs_of_hyperparams = [
                            [0.1,20],
                            [0.1,50],
                            [0.2,20],
                            [0.2,50],
                            [0.3,20],
                            [0.3,50],
                       ]

plot_dbscan_clusters(dataset, pairs_of_hyperparams)

"""

"""
Основные кластеры: Основные кластеры окрашиваются в разные цвета, заданные в списке colors ("black", "#FF5533", "#00B050", "orange", "blue", "purple").
Цвета назначаются каждому кластеру с уникальной меткой.
Шум: Точки, которые считаются шумом (то есть, точки с меткой -1), окрашиваются в черный цвет ("black").
"""

