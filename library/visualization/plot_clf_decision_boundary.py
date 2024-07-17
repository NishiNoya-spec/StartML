import numpy as np
import matplotlib.pyplot as plt

### Для бинарной классификации

def plot_decision_boundary(model, X, y, pixel_step=0.02):
    """
    Визуализирует границы решения модели классификации на двумерном пространстве признаков

    Параметры:
        model : обученная модель классификации
        X : признаки
        y : метки классов
        pixel_step : шаг для генерации точек на плоскости (по умолчанию 0.02)
    """
    # Создание нового графика
    fig = plt.figure(figsize=(16, 10))
    
    # Вычисление границ признакового пространства
    x1_min, x1_max = X.values[:, 0].min() - 1, X.values[:, 0].max() + 1
    x2_min, x2_max = X.values[:, 1].min() - 1, X.values[:, 1].max() + 1

    # Генерация точек на плоскости
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, pixel_step),
        np.arange(x2_min, x2_max, pixel_step)
    )

    # Прогнозирование классов для каждой точки на плоскости
    Z = model.predict(pd.DataFrame(np.c_[xx1.ravel(), xx2.ravel()],
                              columns=["x1", "x2"]))

    Z = Z.reshape(xx1.shape)

    # Заполнение пространства контурами, отображающими прогнозы модели
    cs = plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired)

    plt.axis("tight")

    # Отображение точек данных для каждого класса и обученных областей
    for i, n, c in zip(range(2), model.classes_, ["#FF5533", "#00B050"]):
        idx = np.where(y == i)
        plt.scatter(
            X.values[idx, 0],
            X.values[idx, 1],
            c=c,
            s=20,
            edgecolor="k",
            label="Class %s" % n,
        )
    
    # Установка пределов осей и добавление подписей
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.legend(loc="upper right")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

### *********************

#~ plot_decision_boundary(decision_tree, X, y)

### *********************

### Для нескольких классов 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, pixel_step=0.02):
    """
    Визуализирует границу решения для модели классификации

    Параметры:
        model : обученная модель классификации (дерево решений)
        X : признаки (данные)
        y : метки классов
        pixel_step : шаг для генерации точек на плоскости (по умолчанию 0.02)
    """
    # Создание нового графика
    fig = plt.figure(figsize=(16, 10))
    
    # Вычисление границ признакового пространства
    x1_min, x1_max = X.values[:, 0].min() - 1, X.values[:, 0].max() + 1
    x2_min, x2_max = X.values[:, 1].min() - 1, X.values[:, 1].max() + 1

    # Генерация точек на плоскости
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, pixel_step),
        np.arange(x2_min, x2_max, pixel_step)
    )

    # Прогнозирование классов для каждой точки на плоскости
    Z = model.predict(pd.DataFrame(np.c_[xx1.ravel(), xx2.ravel()],
                              columns=["x1", "x2"]))

    Z = Z.reshape(xx1.shape)

    # Заполнение пространства контурами, отображающими прогнозы модели
    cs = plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired)

    plt.axis("tight")

    # Список цветов для различных классов
    colors = ["#FF5533", "#00B050", "#3370FF", "#FFFF33", "#FF33FF", "#33FFFF", "#FFAA33"]
    
    # Ограничиваем количество цветов до числа классов
    colors = colors[:len(model.classes_)]
    
    # Отображение точек данных для каждого класса и обученных областей
    for i, n, c in zip(range(len(model.classes_)), model.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(
            X.values[idx, 0],
            X.values[idx, 1],
            c=c,
            s=20,
            edgecolor="k",
            label="Class %s" % n,
        )
    
    # Установка пределов осей и добавление подписей
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.legend(loc="upper right")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

### *********************

###^ Функция для построения границ решения от Никиты Табакаева

def plot_surface(clf, X, y):
    plot_step = 0.01
    palette = sns.color_palette(n_colors=len(np.unique(y)))
    cmap = ListedColormap(palette)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=.7,
                edgecolors=np.array(palette)[y], linewidths=2)

