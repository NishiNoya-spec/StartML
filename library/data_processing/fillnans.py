import numpy as np

# 1. Удаление признаков с более чем половиной пропусков (настраиваемый):

class MissingValuesHandler:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.columns_dropped = []

    def drop_columns_with_many_missing_values(self, df):
        """
        Удаляет признаки с более чем указанным порогом пропусков.

        Параметры:
        - df: DataFrame, исходные данные.

        Возвращает:
        - DataFrame без удаленных признаков.
        """
        threshold = self.threshold * df.shape[0]
        self.columns_dropped = [col for col in df.columns if df[col].isna().sum() > threshold]
        return df.drop(columns=self.columns_dropped)

    def get_dropped_columns(self):
        """
        Возвращает список отброшенных признаков.

        Возвращает:
        - Список отброшенных признаков.
        """
        print(f"Кол-во отброшенных признаков: {len(self.columns_dropped)}")
        return self.columns_dropped


# 2. Заполнение пропусков предыдущими значениями:

def fill_missing_with_previous(df):
    """
    Заполняет пропуски предыдущими значениями.

    Параметры:
    - df: DataFrame, исходные данные.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    return df.fillna(method='ffill')

# 3. Заполнение пропусков, смотря на похожие объекты:

def fill_missing_with_grouped_mean(df, group_column, target_column):
    """
    Заполняет пропуски средним значением из группы похожих объектов.

    Параметры:
    - df: DataFrame, исходные данные.
    - group_column: str, название столбца, по которому будет производиться группировка.
    - target_column: str, название столбца с пропущенными значениями.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    grouped_means = df.groupby(group_column)[target_column].transform('mean')
    return df[target_column].fillna(grouped_means)

# 4. Заполнение пропусков средним значением:

def fill_missing_with_mean(df, column):
    """
    Заполняет пропуски средним значением.

    Параметры:
    - df: DataFrame, исходные данные.
    - column: str, название столбца с пропущенными значениями.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    mean = df[column].mean()
    return df[column].fillna(mean)

# 5. Заполнение пропусков медианой:

def fill_missing_with_median(df, column):
    """
    Заполняет пропуски медианой.

    Параметры:
    - df: DataFrame, исходные данные.
    - column: str, название столбца с пропущенными значениями.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    median = df[column].median()
    return df[column].fillna(median)

# 6. Интерполяция между соседними значениями:

def fill_missing_with_interpolation(df, column):
    """
    Заполняет пропуски линейной интерполяцией между соседними значениями.

    Параметры:
    - df: DataFrame, исходные данные.
    - column: str, название столбца с пропущенными значениями.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    return df[column].interpolate(method='linear')

# 7. Заполнение пропусков случайными значениями из распределения признака:

def fill_missing_with_random_values(df, column):
    """
    Заполняет пропуски случайными значениями из распределения признака.

    Параметры:
    - df: DataFrame, исходные данные.
    - column: str, название столбца с пропущенными значениями.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    mean = df[column].mean()
    std = df[column].std()
    mask = df[column].isnull()
    df.loc[mask, column] = np.random.normal(mean, std, size=mask.sum())
    return df

# 8. Использование модели машинного обучения для заполнения пропусков

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def fill_missing_with_model(data_filled, target_column):
    """
    Заполняет пропуски с использованием модели обучения.

    Параметры:
    - data_filled: DataFrame, данные с пропусками.
    - target_column: str, название столбца, в котором нужно заполнить пропуски.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    # Создаем копию данных для заполнения пропусков
    data_filled = data_filled.copy()

    # Разделяем данные на набор с пропусками и без пропусков
    data_with_missing = data_filled[data_filled.isnull().any(axis=1)]
    data_without_missing = data_filled.dropna()

    # Выбираем признаки для обучения модели
    X_train = data_without_missing.drop(target_column, axis=1)
    y_train = data_without_missing[target_column]

    # Обучаем модель на данных без пропусков
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Предсказываем значения для пропущенных данных
    X_missing = data_with_missing.drop(target_column, axis=1)
    predictions = model.predict(X_missing)

    # Заполняем пропущенные значения предсказанными
    data_filled.loc[data_with_missing.index, target_column] = predictions
    
    return data_filled

# 9. Заполнение пропусков самым популярным значением

def fill_categorical_missing_with_most_common(df, categorical_columns):
    """
    Заполняет пропуски в категориальных признаках самыми популярными значениями.

    Параметры:
    - df: DataFrame, исходные данные.
    - categorical_columns: list, список категориальных признаков.

    Возвращает:
    - DataFrame с заполненными пропусками.
    """
    for col in categorical_columns:
        most_common = df[col].mode()[0]  # Находим самое популярное значение
        df[col].fillna(most_common, inplace=True)  # Заполняем пропуски этим значением
    return df
