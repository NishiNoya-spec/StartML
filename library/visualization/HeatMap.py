import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('E:/Рабочий стол С/python_projects/carpov_courses_python')

from library.visualization.visual_preset import set_visualization_settings
set_visualization_settings()

def plot_correlation_heatmap(df, sns_set=False):
    """
    Визуализирует тепловую карту корреляции между числовыми признаками.

    Параметры:
    - df: DataFrame, исходные данные.
    - numeric_columns: list, список числовых признаков.
    - sns_set: boolean, включение графических sns преднастроек для графика.
    """
    if sns_set:
        sns.set()
    else:
        set_visualization_settings()
    numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
    cm = np.corrcoef(df[numeric_columns].values.T)
    fig = plt.figure(figsize=(10, 8))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=numeric_columns, xticklabels=numeric_columns)
    plt.title('Features Correlation Heatmap')
    plt.show()

# Пример использования:

# Построение тепловой карты корреляции
# plot_correlation_heatmap(df, numeric_columns)

### *********************

#~ from library.visualization.HeatMap import plot_correlation_heatmap

#~ plot_correlation_heatmap(df, numeric_columns)

### *********************

#! ==============================================================>>

def plot_scatter_matrix(df, target_column, alpha=0.2, figsize=(20, 20), diagonal='kde'):
    """
    Создает матрицу рассеяния для отображения взаимосвязей и распределения между признаками и целевым столбцом.

    Параметры:
    - df: DataFrame, исходные данные.
    - features: list, список признаков для анализа.
    - target_column: str, имя целевого признака.
    - alpha: float, прозрачность точек на графике (по умолчанию 0.2).
    - figsize: tuple, размер фигуры (по умолчанию (20, 20)).
    - diagonal: str, тип диагонали ('kde' для графика плотности, 'hist' для гистограммы, None для отсутствия диагонали, по умолчанию 'kde').
    """
    sns.set()
    numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
    sns.pairplot(df, vars=numeric_columns, hue=target_column, diag_kind=diagonal, plot_kws={'alpha': alpha}, height=2.5)
    plt.figure(figsize=figsize)
    plt.show()

# Пример использования:

# plot_scatter_matrix(df, 'target_column')
