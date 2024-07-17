import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_circles

def generate_circles_data(n_samples=1000, noise=0.1, factor=0.4, random_state=0):
    """
    Генерирует синтетические данные в форме двух сфер.

    Возвращает:
        DataFrame
            Данные в формате DataFrame с признаками "x1" и "x2" и целевой переменной "y".
    """

    dataset = make_circles(n_samples=1000,
                        noise=0.1,
                        factor=0.4,
                        random_state=0)

    dataset = pd.DataFrame(np.hstack((dataset[0], dataset[1].reshape(-1, 1))),
                        columns=["x1", "x2", "y"])

    return dataset

def plot_circles_data(dataset):
    """
    Визуализирует синтетические данные в форме двух сфер.

    Параметры:
        dataset : DataFrame
            Данные в формате DataFrame с признаками "x1" и "x2" и целевой переменной "y".
    """
    # Создание нового графика
    fig = plt.figure(figsize=(10, 6))

    # Рассеяние данных с использованием seaborn
    sns.scatterplot(
        x="x1",
        y="x2",
        hue="y",
        edgecolor="k",
        data=dataset
    )

    plt.show()