import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

class CategoricalTransformer:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def simple_encoding(self, cols_to_encode):
        """
        Преобразует категориальные признаки в числовые с помощью one-hot-encoding.

        Параметры:
        - cols_to_encode: list, список имен категориальных признаков.

        Возвращает:
        - DataFrame с преобразованными признаками.
        """
        for col in cols_to_encode:
            if self.df[col].nunique() > 5:
                mean_grouped = self.df.groupby(col)[self.target].mean()
                self.df[col] = self.df[col].map(mean_grouped)
            else:
                one_hot = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                one_hot.replace({False: 0, True: 1}, inplace=True)
                self.df = pd.concat((self.df.drop(col, axis=1), one_hot), axis=1)
        return self.df

#! ==========================================================>>

    def count_encoding(self, cols_to_encode, add_noise=False):
        """
        Преобразует категориальные признаки в числовые с помощью счетчиков.

        Параметры:
        - cols_to_encode: list, список имен категориальных признаков.
        - add_noise: bool, определяет, добавлять ли шум к счетчикам для предотвращения переобучения.

        Возвращает:
        - DataFrame с преобразованными признаками.

        Класс разбивает данные на 5 фолдов с использованием кросс-валидации KFold.
        Для каждого категориального признака из списка cols_to_encode создается новый числовой признак с суффиксом _count_encoded.
        Для каждого фолда вычисляется среднее значение таргетной переменной для каждой категории категориального признака (count_map).
        Для каждого наблюдения в валидационном наборе данных (kf_val) значение нового числового
        признака заполняется средним значением таргетной переменной, соответствующим категории в текущем фолде.
        Если параметр add_noise установлен в True, к новому числовому признаку добавляется шум в диапазоне от 0.95 до 1.05.
        Функция возвращает исходный DataFrame с добавленными числовыми признаками.

        В данной реализации используется кросс-валидация, которая обеспечивает взаимодействие между фолдами. Вот как это происходит:
        1. Данные разбиваются на фолды, и каждый фолд используется как валидационный набор данных для оценки модели,
        обученной на остальных фолдах (тренировочных данных).
        2. Для вычисления счетчиков каждый фолд используется как часть обучающего набора данных, а другие фолды (кроме текущего)
        используются для вычисления статистики (среднего значения таргетной переменной для каждой категории).
        3. Это означает, что информация из других фолдов используется для создания счетчиков в текущем фолде.
        Таким образом, категориальные признаки в одном фолде расчитываются по значениям таргетной переменной из других фолдов.

        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for col in cols_to_encode:
            new_col = col + '_count_encoded'
            self.df[new_col] = 0
            for train_index, val_index in kf.split(self.df):
                kf_train, kf_val = self.df.iloc[train_index], self.df.iloc[val_index]
                count_map = kf_train.groupby(col)[self.target].mean()
                self.df.loc[val_index, new_col] = kf_val[col].map(count_map)
            if add_noise:
                self.df[new_col] *= np.random.uniform(0.95, 1.05, size=(len(self.df),))
        return self.df
    
#! ==========================================================>>

def encode_multiclass_target(df, category, target):
    """
    Кодирует мультиклассовую целевую переменную с использованием счетчиков.

    Параметры:
    - df: DataFrame, исходные данные.
    - category: str, имя категориального признака.
    - target: str, имя целевого признака.

    Возвращает:
    - DataFrame с закодированными мультиклассовыми счетчиками.
    """
    # Создаем копию DataFrame с необходимыми признаками
    df = df[[category, target]].copy()

    # Создаем фиктивные переменные для целевой переменной
    target_dummies = pd.get_dummies(df[target], prefix=category, drop_first=True)
    
    # Объединяем DataFrame с фиктивными переменными
    df = pd.concat((df, target_dummies), axis=1)

    # Для каждой фиктивной переменной вычисляем среднее значение в каждой категории
    for tg in target_dummies.columns:
        df[tg] = df.groupby(category)[tg].transform("mean")

    # Удаляем из итогового DataFrame исходный категориальный признак
    return df.drop(category, axis=1)

#! ==========================================================>>

def process_categorical_columns(df, categorical_columns, target):
    """
    Обрабатывает категориальные признаки DataFrame.

    Параметры:
    - df: DataFrame, исходные данные.
    - categorical_columns: list, список категориальных признаков.

    Возвращает:
    - DataFrame с обработанными категориальными признаками.
    """
    for col in categorical_columns:
        if df[col].nunique() < 4:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            one_hot.replace({False: 0, True: 1}, inplace=True)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)
        else:
            mean_target = encode_multiclass_target(df, col, target).drop(target, axis=1)
            df = pd.concat((df.drop(col, axis=1), mean_target), axis=1)
    return df
