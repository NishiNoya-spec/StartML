from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def plot_pr_curve(model, X, Y, title='Precision-Recall Curve'):
    """
    Строит PR-кривую для модели и расчитывает площадь под ней.

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

    # Получение данных PR-кривой
    precision, recall, _ = precision_recall_curve(Y, y_probas)

    # Построение PR-кривой с настройками внешнего вида
    plt.figure(figsize=(8, 6))  # Размер графика
    plt.plot(recall, precision, color='blue', lw=2, label=f"PR curve: {auc(recall, precision):.3f}")  # Цвет, ширина линии и название кривой
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--') # Добавление диагональной пунктирной линии
    plt.xlim([0.0, 1.0])  # Пределы оси x
    plt.ylim([0.0, 1.05])  # Пределы оси y
    plt.xlabel('Recall')  # Подпись оси x
    plt.ylabel('Precision')  # Подпись оси y
    plt.title(title)  # Название графика
    plt.grid(True, alpha=0.1, linestyle="--")  # Включение сетки
    plt.legend(loc="lower left")  # Расположение легенды

    # Закраска площади под PR-кривой
    plt.fill_between(recall, precision, color='purple', alpha=0.3)

    # Отображение графика
    plt.show()

#! ==============================================================>>

class MultiClassPRCurve:
    def __init__(self, model, X, y):
        """
        Класс для построения PR-кривых для всех классов модели.

        Параметры:
        - model: обученная модель для оценки.
        - X: массив признаков выборки.
        - y: массив целевых значений выборки.
        """
        self.model = model
        self.X = X
        self.y = y

    def plot(self, title='Precision-Recall Curves for All Classes'):
        """
        Построение PR-кривых для всех классов модели и расчет площадей под ними.
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

        # Инициализация списка для хранения площадей под PR-кривыми
        aucs = []

        # Инициализация графика
        plt.figure(figsize=(8, 6))

        # Итерация по каждому классу
        for class_index, class_label in enumerate(classes):
            # Получение индексов текущего класса в y
            class_indices = (self.y == class_label).astype(int)
            # Получение данных PR-кривой для текущего класса
            precision, recall, thresholds = precision_recall_curve(class_indices, y_probas[:, class_index])
            # Построение PR-кривой
            plt.plot(recall, precision, lw=2, label=f"PR curve class {class_label} (AUC = {auc(recall, precision):.3f})")
            # Расчет и сохранение площади под PR-кривой
            aucs.append(auc(recall, precision))
            # Закраска площади под ROC-кривой
            plt.fill_between(recall, precision, alpha=0.3, cmap='viridis')

        # Добавление диагональной пунктирной линии
        plt.plot([0, 1], [1, 0], color='gray', lw=1, linestyle='--')

        # Настройка внешнего вида графика
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.1, linestyle='--')
        plt.legend(loc="lower left")

        # Отображение графика
        plt.show()

        return aucs
    

### *********************

#~ from library.visualization.PR_AUC import MultiClassPRCurve
#~ plotter = MultiClassPRCurve(pipe, X_test, Y_test)
#~ plotter.plot(title=pipe.steps[1][1])

### *********************

#! ==============================================================>>