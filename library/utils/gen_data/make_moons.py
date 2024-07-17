import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

def generate_moons_data(n_samples=1000, noise=0.5, random_state=None):
    """
    Генерирует синтетические данные в форме двух полумесяцев.

    Параметры:
        n_samples : int, необязательный
            Число образцов (по умолчанию 1000).
        noise : float, необязательный
            Уровень шума в данных (по умолчанию 0.5).
        random_state : int или RandomState, необязательный
            Состояние генератора случайных чисел (по умолчанию None).

    Возвращает:
        DataFrame
            Данные в формате DataFrame с признаками "x1" и "x2" и целевой переменной "y".
    """
    # Генерация данных с помощью make_moons
    dataset = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Преобразование данных в DataFrame
    dataset = pd.DataFrame(np.hstack((dataset[0], dataset[1].reshape(-1, 1))),
                           columns=["x1", "x2", "y"])

    return dataset

def plot_moons_data(dataset):
    """
    Визуализирует синтетические данные в форме двух полумесяцев.

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

