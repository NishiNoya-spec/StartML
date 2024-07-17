import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# Инициализация кластеров
inits = [
    'random', ### случайная инициализация, при которой начальные центроиды кластеров выбираются случайным образом из набора данных
    'k-means++', ### улучшенная инициализация, которая выбирает начальные центроиды таким образом, чтобы уменьшить вероятность плохой сходимости
    np.array([[1, 1], [0, 0]]), ### Ручное задание начальных точек (для двух кластеров)
    np.array([[1, -2], [1, -2]])
]

for index, init in enumerate(inits):

    ### Обучение
    kmeans = KMeans(n_clusters=2, init=init, random_state=0).fit(X)
    ### Предсказание 
    y_kmeans = kmeans.predict(X)


# Пример вывода центров кластеров
print("Cluster Centers:\n", kmeans.cluster_centers_)
# Пример вывода количества итераций до сходимости
print("Number of iterations:", kmeans.n_iter_)
