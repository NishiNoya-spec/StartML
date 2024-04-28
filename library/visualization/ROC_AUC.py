from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def plot_roc_curve(model, X, Y, title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Строит ROC-кривую для модели и расчитывает площадь под ней.

    Параметры:
    - model: обученная модель для оценки.
    - X: массив признаков выборки.
    - Y: массив целевых значений выборки.
    """

    if hasattr(model, 'predict_proba'):
        # Если доступен метод predict_proba, используем его для получения вероятностей
        y_probas = model.predict_proba(X)[:, 1]
    else:
        # Иначе получаем отступы и применяем сигмоидную функцию для получения вероятностей
        M = model.decision_function(X)
        y_probas = 1 / (1 + np.exp(-M))

    # Получение данных ROC-кривой
    fpr, tpr, thresholds = roc_curve(Y, y_probas)

    # Построение ROC-кривой с настройками внешнего вида
    plt.figure(figsize=(8, 6))  # Размер графика
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve: {auc(fpr, tpr):.3f}")  # Цвет, ширина линии и название кривой
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Добавление диагональной пунктирной линии
    plt.xlim([0.0, 1.0])  # Пределы оси x
    plt.ylim([0.0, 1.05])  # Пределы оси y
    plt.xlabel('False Positive Rate')  # Подпись оси x
    plt.ylabel('True Positive Rate')  # Подпись оси y
    plt.title(title)  # Название графика
    plt.grid(True, alpha=0.1, linestyle="--")  # Включение сетки
    plt.legend(loc="lower right")  # Расположение легенды

    # Закраска площади под ROC-кривой
    plt.fill_between(fpr, tpr, color='purple', alpha=0.3)

    # Отображение графика
    plt.show()

#! ==============================================================>>

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

class MultiClassROCCurve:
    def __init__(self, model, X, y):
        """
        Класс для построения ROC-кривых для всех классов модели.

        Параметры:
        - model: обученная модель для оценки.
        - X: массив признаков выборки.
        - y: массив целевых значений выборки.
        """
        self.model = model
        self.X = X
        self.y = y

    def plot(self, title='Receiver Operating Characteristic (ROC) Curves for All Classes'):
        """
        Построение ROC-кривых для всех классов модели и расчет площадей под ними.
        """
        # Проверяем, доступен ли метод predict_proba
        if hasattr(self.model, 'predict_proba'):
            # Получение вероятностей принадлежности к классам
            y_probas = self.model.predict_proba(self.X)
        else:
            # Получение отступов от гиперплоскости для каждого класса
            M = self.model.decision_function(self.X)
            y_probas = 1 / (1 + np.exp(-M)) # получение вероятностей через сигмоидную функцию активации 

            # Находим минимальное и максимальное значение среди всех вероятностей
            overall_min = np.min(y_probas)
            overall_max = np.max(y_probas)

            # Проверяем, находятся ли вероятности в диапазоне от 0 до 1
            if 0 <= overall_min <= 1 and 0 <= overall_max <= 1:
                print("Вероятности находятся в диапазоне от 0 до 1.")
            else:
                print("Вероятности не находятся в диапазоне от 0 до 1. Необходима нормализация.")

        # Получение уникальных классов модели
        classes = self.model.classes_

        # Инициализация списка для хранения площадей под ROC-кривыми
        aucs = []

        # Инициализация графика
        plt.figure(figsize=(8, 6))

        # Итерация по каждому классу
        for class_index, class_label in enumerate(classes):
            # Получение индексов текущего класса в y
            class_indices = (self.y == class_label).astype(int)
            # Получение данных ROC-кривой для текущего класса
            fpr, tpr, thresholds = roc_curve(class_indices, y_probas[:, class_index])
            # Построение ROC-кривой
            plt.plot(fpr, tpr, lw=2, label=f"ROC curve class {class_label} (AUC = {auc(fpr, tpr):.3f})")
            # Расчет и сохранение площади под ROC-кривой
            aucs.append(auc(fpr, tpr))
            # Закраска площади под ROC-кривой
            plt.fill_between(fpr, tpr, alpha=0.3, cmap='viridis')

        # Добавление диагональной пунктирной линии
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        # Настройка внешнего вида графика
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.grid(True, alpha=0.1, linestyle='--')
        plt.legend(loc="lower right")

        # Отображение графика
        plt.show()

        return aucs

# Пример использования:

# plot_roc_curve(pipe, X_test, Y_test)

### *********************

#~ from library.visualization.ROC_AUC import MultiClassROCCurve
#~ plotter = MultiClassROCCurve(pipe, X_test, Y_test)
#~ plotter.plot(title=pipe.steps[1][1])

### *********************

#! ==============================================================>>

