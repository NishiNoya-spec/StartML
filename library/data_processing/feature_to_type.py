import pandas as pd
import numpy as np

# ==============================================================>>

def feature_to_date(df, feature):
    """
    Преобразует значения столбца в соответствующий тип данных datetime.

    Параметры:
    - df: DataFrame, содержащий столбец.
    - feature: str, наименование столбца.

    Возвращает:
    - None, если преобразование выполнено успешно,
    - Сообщение об ошибке, если произошла ошибка при преобразовании.
    """
    try:
        df[feature] = pd.to_datetime(df[feature])
    except Exception as e:
        print(f"Ошибка: Невозможно преобразовать столбец {feature} в дату. {str(e)}")

# ==============================================================>>

def feature_to_numeric(df, feature, decimal=','):
    """
    Преобразует значения столбца в соответствующий числовой тип данных.

    Параметры:
    - df: DataFrame, содержащий столбец.
    - feature: str, наименование столбца.
    - decimal: str, символ, используемый для разделения десятичных разрядов (по умолчанию ',').

    Возвращает:
    - None, если преобразование выполнено успешно,
    - Сообщение об ошибке, если произошла ошибка при преобразовании.
    """
    try:
        # Преобразование числовых значений
        df[feature] = pd.to_numeric(df[feature].str.replace(',', '.'), errors='coerce')
    except ValueError:
        try:
            # Преобразование целочисленных значений
            df[feature] = df[feature].astype(int)
        except Exception as e:
            print(f"Ошибка: Невозможно преобразовать столбец {feature} в числовой тип данных. {str(e)}")

# ==============================================================>>

def numeric_categorical_columns(dataset):
    """
    Определяет числовые и категориальные признаки в датасете.

    Параметры:
    - dataset: DataFrame, исходный датасет.

    Возвращает:
    - numeric_columns: list, список числовых признаков.
    - categorical_columns: list, список категориальных признаков.
    """
    numeric_columns = dataset.select_dtypes(exclude=[np.object_]).columns.tolist()
    categorical_columns = dataset.select_dtypes(include=[np.object_]).columns.tolist()
    return numeric_columns, categorical_columns

# Пример использования
# numeric_columns, categorical_columns = numeric_categorical_columns(df)