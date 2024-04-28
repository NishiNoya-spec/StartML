"""
Для отбрасывания константных и квазиконстантных признаков используется метод VarianceThreshold
из библиотеки scikit-learn. Этот метод удаляет признаки, чья дисперсия меньше заданного порога.
Формула для расчета дисперсии проста:

Var(X) = Σ (xᵢ - μ)² / n

Где:

Var(X) - дисперсия признака X.
Σ - сумма по всем элементам.
xᵢ - каждое отдельное значение признака.
μ - среднее значение признака.
n - количество значений признака.

"""

from sklearn.feature_selection import VarianceThreshold

def drop_constant_features(dataset, threshold=0):
    """
    Отбрасывает константные и квазиконстантные признаки из датасета.

    Параметры:
    - dataset: DataFrame, исходный датасет.
    - threshold: float, порог дисперсии, ниже которого признаки считаются константными (по умолчанию 0).

    Возвращает:
    - dataset: DataFrame, датасет с отброшенными константными признаками.
    - constant_cols: list, список названий отброшенных признаков.
    """
    cutter = VarianceThreshold(threshold=threshold)
    cutter.fit(dataset)
    selected_features = cutter.get_support()
    constant_cols = dataset.columns[~selected_features].tolist()
    dataset = dataset.drop(columns=constant_cols, axis=1)
    return dataset, constant_cols

# Пример использования
# data, constant_cols = drop_constant_features(data)


"""
В данной конструкции dataset.columns[~selected_features].tolist(),
переменная constant_cols будет содержать только те признаки,
для которых значение в массиве selected_features равно False.
При использовании метода get_support() после обучения VarianceThreshold,
он возвращает булев массив, где True указывает на признаки,
которые не были отфильтрованы (то есть их дисперсия выше порога),
а False указывает на признаки, которые были отфильтрованы.
Использование оператора ~ перед selected_features инвертирует значения в массиве,
то есть True становится False, а False становится True.
Поэтому dataset.columns[~selected_features] вернет только имена признаков,
для которых значение в selected_features равно False, то есть только те признаки,
которые были отфильтрованы.
Таким образом, переменная constant_cols будет содержать имена признаков,
которые были отфильтрованы с использованием VarianceThreshold.

"""