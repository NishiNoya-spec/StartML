import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_calibration_curve(model, X, y, n_bins=10, title='Calibration Curve'):
    """
    Строит калибровочную кривую для каждого класса модели.

    Параметры:
    - model: обученная модель для оценки.
    - X: массив признаков выборки.
    - y: массив целевых значений выборки.
    - n_bins: количество бинов для разбиения.
    - title: заголовок графика.
    """
    # Проверяем, доступен ли метод predict_proba
    if hasattr(model, 'predict_proba'):
        # Получаем вероятности для каждого класса
        prob_pos = model.predict_proba(X)
    else:
        # Получаем отступы от гиперплоскости и применяем сигмоидную функцию для получения вероятностей
        M = model.decision_function(X)
        prob_pos = 1 / (1 + np.exp(-M))

    # Получаем уникальные классы
    classes = np.unique(y)

    # Инициализируем график
    plt.figure(figsize=(8, 6))

    # Итерируемся по каждому классу
    for class_index, class_label in enumerate(classes):
        # Получаем бинарные метки для текущего класса
        y_binary = (y == class_label).astype(int)
        # Получаем калибровочную кривую для текущего класса
        fraction_of_positives, mean_predicted_value = calibration_curve(y_binary, prob_pos[:, class_index], n_bins=n_bins)
        # Построение калибровочной кривой для текущего класса
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'Class {class_label}')

    # Добавляем диагональную пунктирную линию
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Настраиваем внешний вид графика
    plt.title(title)
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True, alpha=0.1, linestyle='--')

    # Отображаем график
    plt.show()

### *********************

#~ from library.visualization.CalibrationCurve import plot_calibration_curve
#~ plot_calibration_curve(pipe, X_test, Y_test)

### *********************