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


#~ ==========================================================>>

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import itertools

class MeanTargetEncoder_(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_cols, numeric_cols, target_name, mte_fillna_strategy="mean", mte_strategy="cv_split", noise_k=0.006):              
    
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_name = target_name
        self.mte_fillna_strategy = mte_fillna_strategy
        self.mte_strategy = mte_strategy
        self.noise_k = noise_k

    def fit(self, X, y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1) 
        
        ### Посчитаем на валидации средние значения таргета
        if self.mte_strategy == "none":

            self.dict_of_means = {}
            for col in self.categorical_cols:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean()


        elif self.mte_strategy == "noise":

            self.dict_of_means = {}
            for col in self.categorical_cols:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean() + self.noise_k * np.random.randn(len(X_with_target[col].unique()))
    

        elif self.mte_strategy == "cv_split":

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.dict_of_means = {}
            for col in self.categorical_cols:
                col_means = {}
                for index, (train_index, val_index) in enumerate(kf.split(X_with_target)):

                    kf_train, kf_val = X_with_target.iloc[train_index], X_with_target.iloc[val_index]
                    ### посчитали на ~4ех трейновых сплитах средние значения таргета для каждой категории
                    count_map = kf_train.groupby(col)[self.target_name].mean()
                    count_map = count_map + self.noise_k * np.random.randn(len(count_map))
                    col_means[f"fold_{index}"] = count_map

                self.dict_of_means[col] = pd.concat(col_means.values(), axis=1).mean(axis=1)


        elif self.mte_strategy == "multiclass":

            ### Запомним все классы
            self.mte_class_names = {col : sorted([f"{col}_{value}" for value in X_with_target[self.target_name].unique()]) for col in self.categorical_cols}


        else:
            raise ValueError("Unsupported mte_strategy strategy. Please choose either 'none', 'noise' or 'cv_split'.")
        
        return self
        

    def transform(self, X, y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        df_ = pd.concat((X_fit, y_fit), axis=1) 

        if self.mte_strategy != "multiclass":

            for col in self.categorical_cols:
                df_[col] = df_[col].map(self.dict_of_means[col])
                
                ### Если на проде или на тестовой выборке мы получили новые категории, которые не были в обучении
                ### то заполняем эти данные по общему среднему всех полученных значений для того чтобы избавится от NaN-ов
                if df_[col].isna().sum() != 0:
                    if self.mte_fillna_strategy == 'mean':
                        mean_value = self.dict_of_means[col].values.mean()
                        df_[col] = df_[col].fillna(mean_value)
                    elif self.mte_fillna_strategy == 'mode':
                        mode_value = df_[col].mode()[0]
                        df_[col] = df_[col].fillna(mode_value)
                    else:
                        raise ValueError("Unsupported fillna strategy. Please choose either 'mean' or 'mode'.")

        else:

            target_dummies_cols = [] # Сюда будем записывать все столбцы, которые будут закодированы 

            for col in self.categorical_cols:

                # Создаем фиктивные переменные для целевой переменной
                target_dummies = pd.get_dummies(df_[self.target_name], prefix=col)
            
                ### Фиксируем полученные столбцы после кодирования
                target_dummies_cols.extend(target_dummies.columns)

                # Объединяем DataFrame с фиктивными переменными
                df_ = pd.concat((df_, target_dummies), axis=1)

                # Для каждой фиктивной переменной вычисляем среднее значение в каждой категории
                for tg in target_dummies.columns:
                    df_[tg] = df_.groupby(col)[tg].transform("mean")

                # Удаляем из итогового DataFrame исходный категориальный признак
                df_ = df_.drop(col, axis=1)

            # Получение списка всех имен столбцов по классам
            all_mte = list(itertools.chain(*list(self.mte_class_names.values())))

            ### На проде или на тесте мы можем не получить классы, которые были в обучении,
            ### в таком случае нам необходимо зафиксировать данные классы и создать столбцы для 
            ### данных классов, заполнив их нулями, таким образом мы сохраним изначальную размерность датасета на обучении
            missing_columns = [
                x 
                for x in all_mte
                if x not in df_.columns
                and
                x not in self.numeric_cols
            ]

            ### Отсутствующие классы
            ### необходимо добавить: заполним их просто нулями
            
            if len(missing_columns) != 0:

                zeros = np.zeros((df_.shape[0], len(missing_columns)))
                zeros = pd.DataFrame(zeros,
                                    columns=missing_columns,
                                    index=df_.index)

                df_ = pd.concat((df_, zeros), axis=1)


            ### Также на проде или на тесте мы можем получить новые классы, которые не видлеи на обучении,
            ### таким образом мы можем получить новые столбцы, делать этого нельзя, так как мы потеряем нашу размерноть, 
            ### поэтому фиксируем их и отбрасываем
            extra_columns = [
                x
                for x in target_dummies_cols
                if x not in all_mte
            ]

            ### Новые классы необходимо убрать
            df_ = df_.drop(extra_columns, axis=1)


        return df_[sorted(df_.columns)]

#^ ==========================================================>>

import itertools
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class OneHotEncoder_(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_cols, numeric_cols, target_name, encoding_method="one_hot"):              
    
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_name = target_name
        self.encoding_method = encoding_method
        self.label_encoding_dicts = {}  # Добавление атрибута для хранения словарей кодирования

    def fit(self, X, y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1) 
        
        if self.encoding_method == 'one_hot':

            ### Запомним все ohe колонки и их названия
            self.ohe_names = {col : sorted([f"{col}_{value}" for value in X_with_target[col].unique()]) for col in self.categorical_cols}


        elif self.encoding_method == 'label':
            ### Закодируем каждое категориальное значение в категориальном признаке  
            for col in self.categorical_cols:
                label_encoding_dicts = {val: idx for idx, val in enumerate(X_with_target[col].unique())}
                self.label_encoding_dicts[col] = label_encoding_dicts


        return self
        
    def transform(self, X, y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        df_ = pd.concat((X_fit, y_fit), axis=1) 

        if self.encoding_method == 'one_hot':
            data_part = pd.get_dummies(df_[self.categorical_cols], prefix=None)
            data_part.replace({False: 0, True: 1}, inplace=True)

            # Дропаем все оригинальные категориальные колонки, так как они нам больше не нужны 
            df_ = df_.drop(self.categorical_cols, axis=1)
            # Присоединяем все преобразованные категориальные колонки к датасету 
            df_ = pd.concat((df_, data_part), axis=1)

            # Получение списка всех имен столбцов one-hot в рамках данного класса 
            all_ohe = list(itertools.chain(*list(self.ohe_names.values())))

            ### Фиксируем полученные столбцы после кодирования ohe
            data_part_cols = data_part.columns

            ### На проде или на тесте мы можем не получить категории, которые были в обучении,
            ### в таком случае нам необходимо зафиксировать данные категории и создать столбцы для 
            ### данных категорий, заполнив их нулями, таким образом мы сохраним изначальную размерность датасета на обучении
            missing_columns = [
                x 
                for x in all_ohe
                if x not in df_.columns
                and
                x not in self.numeric_cols
            ]

            ### Отсутствующие категории (бинарные колонки)
            ### необходимо добавить: заполним их просто нулями
            
            if len(missing_columns) != 0:

                zeros = np.zeros((df_.shape[0], len(missing_columns)))
                zeros = pd.DataFrame(zeros,
                                    columns=missing_columns,
                                    index=df_.index)

                df_ = pd.concat((df_, zeros), axis=1)


            ### Также на проде или на тесте мы можем получить новые категории, которые не видлеи на обучении,
            ### таким образом мы можем получить новые столбцы, делать этого нельзя, так как мы потеряем нашу размерноть, 
            ### поэтому фиксируем их и отбрасываем
            extra_columns = [
                x
                for x in data_part_cols
                if x not in all_ohe
            ]

            ### Новые категории необходимо убрать
            df_ = df_.drop(extra_columns, axis=1)

            # Инициализируем новый DataFrame для хранения результата
            new_df = pd.DataFrame()

            # Удаляем первый столбец для каждой категории для избегания мультиколлинеарности 
            for unique_pref in self.ohe_names.keys():
                filtered_cols = df_.filter(like=unique_pref).columns
                # Получаем все столбцы категории
                filtered_df = df_[filtered_cols]
                # Удаляем первый столбец и добавляем остальные в новый DataFrame
                new_df = pd.concat([new_df, filtered_df.drop(filtered_df.columns[0], axis=1)], axis=1)

            # Обновляем df_ новыми значениями без первого столбца для каждой категории
            df_ = new_df

        elif self.encoding_method == 'label':

            for col in self.categorical_cols:
                encoding_dict = self.label_encoding_dicts[col]
                df_[col] = df_[col].map(encoding_dict)
                ### Если на проде или на тестовой выборке мы получили новые категории,
                # то заполняем неизвестны категории значением -1
                if df_[col].isna().sum() != 0:
                    df_[col] = df_[col].fillna(-1)

        else:
            raise ValueError("Unsupported encoding method. Please choose either 'one_hot' or 'label'.")
        
            
        return df_[sorted(df_.columns)]

#~ ==========================================================>>