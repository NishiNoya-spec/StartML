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
    
    def __init__(self, categorical_cols, numeric_cols, target_name, mte_fillna_strategy="none", mte_strategy="noise", noise_k=0.006):              
        self.categorical_cols = categorical_cols ### имена столбцов с категориальными значениями
        self.numeric_cols = numeric_cols ### имена столбцов с вещественными значениями
        self.target_name = target_name ### имя целевой переменной
        self.mte_fillna_strategy = mte_fillna_strategy ### none; mean; mode
        self.mte_strategy = mte_strategy ### none; noise; cv_split; multiclass
        self.noise_k = noise_k ### коэффициент шума
        self.mean_target_dict = {} ### словарь для хранения средних значений по таргету (классификация) сгруппированных категориальных значений 
        self.dict_of_means = {} ### словарь для хранения средних значений по таргету (регрессия) сгруппированных категориальных значений 

    def fit(self, X, y):
        X_fit = X.copy()
        y_fit = y.copy()
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1) 
        
        if self.mte_strategy == "none":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе вещественной-целевой переменной.
            """

            for col in self.categorical_cols:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean()

        elif self.mte_strategy == "noise":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе вещественной-целевой переменной и добавляет шум.
            """

            for col in self.categorical_cols:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean() + self.noise_k * np.random.randn(len(X_with_target[col].unique()))
    
        elif self.mte_strategy == "cv_split":

            """
            В данной реализации используется кросс-валидация, которая обеспечивает взаимодействие между фолдами. Вот как это происходит:
            1. Данные разбиваются на фолды, и каждый фолд используется как валидационный набор данных для оценки модели,
            обученной на остальных фолдах (тренировочных данных).
            2. Для вычисления счетчиков каждый фолд используется как часть обучающего набора данных, а другие фолды (кроме текущего)
            используются для вычисления статистики (среднего значения таргетной переменной для каждой категории).
            3. Это означает, что информация из других фолдов используется для создания счетчиков в текущем фолде.
            Таким образом, категориальные признаки в одном фолде расчитываются по значениям таргетной переменной из других фолдов.
            """

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for col in self.categorical_cols:
                col_means = []
                for index, (train_index, val_index) in enumerate(kf.split(X_with_target)):
                    kf_train = X_with_target.iloc[train_index]
                    count_map = kf_train.groupby(col)[self.target_name].mean()
                    count_map = count_map + self.noise_k * np.random.randn(len(count_map))
                    col_means.append(count_map)
                self.dict_of_means[col] = pd.concat(col_means, axis=1).mean(axis=1)

        elif self.mte_strategy == "multiclass":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе мультиклассовой целевой переменной.
            """

            self.target_dummies = pd.get_dummies(y, prefix=self.target_name)
            X_with_target = pd.concat([X, self.target_dummies], axis=1)
            
            for col in self.categorical_cols:
                for tg in self.target_dummies.columns:
                    mean_target = X_with_target.groupby(col)[tg].mean()
                    if tg not in self.mean_target_dict:
                        self.mean_target_dict[tg] = {}
                    self.mean_target_dict[tg][col] = mean_target

            print(f"Сводка для кодирования мультиклассового-счетчика: ")
            print(f"Кол-во категориальных фичей: {len(self.categorical_cols)}")
            print(f"Кол-во уникальных классов: {y_fit.nunique()}")
            print(f"Кол-во столбцов, которое должно получится при кодировании: {len(self.categorical_cols) * y_fit.nunique()}")
            print(f"Окончательное кол-во столбцов с учетом отброса по одному столбцу от закодированного класса получается: {(len(self.categorical_cols) * y_fit.nunique()) + (X.shape[1] - len(self.categorical_cols)) - len(self.categorical_cols)}")

        else:
            raise ValueError("Unsupported mte_strategy strategy. Please choose either 'none', 'noise', 'cv_split' or 'multiclass'.")
        
        return self

    def transform(self, X):
        X_trans = X.copy()

        if self.mte_strategy != "multiclass":
            for col in self.categorical_cols:
                X_trans[col] = X_trans[col].map(self.dict_of_means[col])
                X_trans[col] = self.fillnans(X_trans, col, self.dict_of_means[col])
        else:
            new_columns = {}
            for col in self.categorical_cols:
                for target in self.mean_target_dict.keys():
                    new_col_name = f"{col}_{target}"
                    new_columns[new_col_name] = X[col].map(self.mean_target_dict[target][col])
                    new_columns[new_col_name] = self.fillnans(new_columns, new_col_name, self.mean_target_dict[target][col])

                ### Удаляем один закодированный столбец по категории для того чтобы избежать мультиколлинеарности
                del new_columns[f"{col}_{target}"]

            X_trans = X_trans.join(pd.DataFrame(new_columns))
            X_trans.drop(columns=self.categorical_cols, inplace=True)

        return X_trans
    
    def fillnans(self, X, col, dict_of_means):

        ### Если на проде или на тестовой выборке мы получили новые категории, которые не были в обучении
        ### то заполняем эти данные по общему среднему всех полученных значений для того чтобы избавится от NaN-ов

        if X[col].isna().sum() != 0:
            if self.mte_fillna_strategy == "none":
                X[col] = X[col].fillna(0)
            elif self.mte_fillna_strategy == 'mean':
                mean_value = dict_of_means.mean()
                X[col] = X[col].fillna(mean_value)
            elif self.mte_fillna_strategy == 'mode':
                mode_value = X[col].mode()[0]
                X[col] = X[col].fillna(mode_value)
            else:
                raise ValueError("Unsupported fillna strategy. Please choose either 'none', 'mean' or 'mode'.")
        return X[col]




#^ ==========================================================>>




class OneHotEncoder_(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_cols, numeric_cols, target_name, encoding_method="one_hot"):              

        self.categorical_cols = categorical_cols ### имена столбцов с категориальными значениями
        self.numeric_cols = numeric_cols ### имена столбцов с вещественными значениями
        self.target_name = target_name ### имя целевой переменной
        self.encoding_method = encoding_method ### one_hot; label
        self.label_encoding_dicts = {}  ### словарь для хранения значений кодирования по label-енкодеру 

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
        
    def transform(self, X):
        
        df_ = X.copy()

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
                # то заполняем неизвестные категории значением -1
                if df_[col].isna().sum() != 0:
                    df_[col] = df_[col].fillna(-1)

        else:
            raise ValueError("Unsupported encoding method. Please choose either 'one_hot' or 'label'.")
        
            
        return df_[sorted(df_.columns)]







#~ ==========================================================>>






class EncoderTransformer_(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_cols, numeric_cols, target_name, threshold=5, encoding_method="one_hot", mte_strategy="noise", mte_fillna_strategy="none", noise_k=0.006):              

        self.categorical_cols = categorical_cols ### имена столбцов с категориальными значениями
        self.numeric_cols = numeric_cols ### имена столбцов с вещественными значениями
        self.target_name = target_name ### имя целевой переменной
        self.threshold = threshold ### порог уникальных значений категориальной фичи, при которой определяется будет она закодирована по ohe или mean target 
        self.encoding_method = encoding_method ### one_hot; label
        self.label_encoding_dicts = {}  ### словарь для хранения значений кодирования по label-енкодеру 
        self.mte_fillna_strategy = mte_fillna_strategy ### none; mean; mode
        self.mte_strategy = mte_strategy ### none; noise; cv_split; multiclass
        self.noise_k = noise_k ### коэффициент шума
        self.mean_target_dict = {} ### словарь для хранения средних значений по таргету (классификация) сгруппированных категориальных значений 
        self.dict_of_means = {} ### словарь для хранения средних значений по таргету (регрессия) сгруппированных категориальных значений 
        self.cols_to_ohe_encode = [] ### колонки, которые будут закодированы ohe-методами
        self.cols_to_mean_target_encode = [] ### колонки, которые будут закодированы mean tearget-методами


    def fit(self, X, y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1) 
        
        ### Распределение категориальных фичей по энкодерам 
        for col in self.categorical_cols:
            if X_with_target[col].nunique() > self.threshold:
                self.cols_to_mean_target_encode.append(col)
            else:
                self.cols_to_ohe_encode.append(col)

        print(f"При отсечке = {self.threshold}")
        print(f"{len(self.cols_to_mean_target_encode)} - Кол-во фичей, которое будет закодировано mean target-энкодером")
        print(f"{len(self.cols_to_ohe_encode)} - Кол-во фичей, которое будет закодировано ohe-энкодером")

        ### Реализация ohe-энкодеров

        if self.encoding_method == 'one_hot':

            ### Запомним все ohe колонки и их названия
            self.ohe_names = {col : sorted([f"{col}_{value}" for value in X_with_target[col].unique()]) for col in self.cols_to_ohe_encode}


        elif self.encoding_method == 'label':
            ### Закодируем каждое категориальное значение в категориальном признаке  
            for col in self.cols_to_ohe_encode:
                label_encoding_dicts = {val: idx for idx, val in enumerate(X_with_target[col].unique())}
                self.label_encoding_dicts[col] = label_encoding_dicts


        ### Реализация mean target-энкодеров

        if self.mte_strategy == "none":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе вещественной-целевой переменной.
            """

            for col in self.cols_to_mean_target_encode:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean()

        elif self.mte_strategy == "noise":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе вещественной-целевой переменной и добавляет шум.
            """

            for col in self.cols_to_mean_target_encode:
                self.dict_of_means[col] = X_with_target.groupby(col)[self.target_name].mean() + self.noise_k * np.random.randn(len(X_with_target[col].unique()))
    
        elif self.mte_strategy == "cv_split":

            """
            В данной реализации используется кросс-валидация, которая обеспечивает взаимодействие между фолдами. Вот как это происходит:
            1. Данные разбиваются на фолды, и каждый фолд используется как валидационный набор данных для оценки модели,
            обученной на остальных фолдах (тренировочных данных).
            2. Для вычисления счетчиков каждый фолд используется как часть обучающего набора данных, а другие фолды (кроме текущего)
            используются для вычисления статистики (среднего значения таргетной переменной для каждой категории).
            3. Это означает, что информация из других фолдов используется для создания счетчиков в текущем фолде.
            Таким образом, категориальные признаки в одном фолде расчитываются по значениям таргетной переменной из других фолдов.
            """

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for col in self.cols_to_mean_target_encode:
                col_means = []
                for index, (train_index, val_index) in enumerate(kf.split(X_with_target)):
                    kf_train = X_with_target.iloc[train_index]
                    count_map = kf_train.groupby(col)[self.target_name].mean()
                    count_map = count_map + self.noise_k * np.random.randn(len(count_map))
                    col_means.append(count_map)
                self.dict_of_means[col] = pd.concat(col_means, axis=1).mean(axis=1)

        elif self.mte_strategy == "multiclass":

            """
            Кодирует категориальные признаки с использованием счетчиков на основе мультиклассовой целевой переменной.
            """

            self.target_dummies = pd.get_dummies(y, prefix=self.target_name)
            X_with_target = pd.concat([X, self.target_dummies], axis=1)
            
            for col in self.cols_to_mean_target_encode:
                for tg in self.target_dummies.columns:
                    mean_target = X_with_target.groupby(col)[tg].mean()
                    if tg not in self.mean_target_dict:
                        self.mean_target_dict[tg] = {}
                    self.mean_target_dict[tg][col] = mean_target

            print(f"Сводка для кодирования мультиклассового-счетчика: ")
            print(f"Кол-во категориальных фичей: {len(self.cols_to_mean_target_encode)}")
            print(f"Кол-во уникальных классов: {y_fit.nunique()}")
            print(f"Кол-во столбцов, которое должно получится при кодировании: {len(self.cols_to_mean_target_encode) * y_fit.nunique()}")
            print(f"Окончательное кол-во столбцов с учетом отброса по одному столбцу от закодированного класса получается: {(len(self.cols_to_mean_target_encode) * y_fit.nunique()) + (X.shape[1] - len(self.cols_to_mean_target_encode)) - len(self.cols_to_mean_target_encode)}")

        else:
            raise ValueError("Unsupported mte_strategy strategy. Please choose either 'none', 'noise', 'cv_split' or 'multiclass'.")

        return self
    

            
    def transform(self, X):
        
        X_ = X.copy()
        df_orig = X_.drop(self.cols_to_ohe_encode + self.cols_to_mean_target_encode, axis=1)
        df_ohe = X_[self.cols_to_ohe_encode]
        df_mte = X_[self.cols_to_mean_target_encode]

        if len(self.cols_to_ohe_encode) != 0:

            if self.encoding_method == 'one_hot':
                data_part = pd.get_dummies(df_ohe, prefix=None)
                data_part.replace({False: 0, True: 1}, inplace=True)

                # Дропаем все оригинальные категориальные колонки, так как они нам больше не нужны 
                df_ohe = df_ohe.drop(self.cols_to_ohe_encode, axis=1)
                # Присоединяем все преобразованные категориальные колонки к датасету 
                df_ohe = pd.concat((df_ohe, data_part), axis=1)

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
                    if x not in df_ohe.columns
                    and
                    x not in self.numeric_cols
                ]

                ### Отсутствующие категории (бинарные колонки)
                ### необходимо добавить: заполним их просто нулями
                
                if len(missing_columns) != 0:

                    zeros = np.zeros((df_ohe.shape[0], len(missing_columns)))
                    zeros = pd.DataFrame(zeros,
                                        columns=missing_columns,
                                        index=df_ohe.index)

                    df_ohe = pd.concat((df_ohe, zeros), axis=1)


                ### Также на проде или на тесте мы можем получить новые категории, которые не видлеи на обучении,
                ### таким образом мы можем получить новые столбцы, делать этого нельзя, так как мы потеряем нашу размерноть, 
                ### поэтому фиксируем их и отбрасываем
                extra_columns = [
                    x
                    for x in data_part_cols
                    if x not in all_ohe
                ]

                ### Новые категории необходимо убрать
                df_ohe = df_ohe.drop(extra_columns, axis=1)

                # Инициализируем новый DataFrame для хранения результата
                new_df = pd.DataFrame()

                # Удаляем первый столбец для каждой категории для избегания мультиколлинеарности 
                for unique_pref in self.ohe_names.keys():
                    filtered_cols = df_ohe.filter(like=unique_pref).columns
                    # Получаем все столбцы категории
                    filtered_df = df_ohe[filtered_cols]
                    # Удаляем первый столбец и добавляем остальные в новый DataFrame
                    new_df = pd.concat([new_df, filtered_df.drop(filtered_df.columns[0], axis=1)], axis=1)

                # Обновляем df_ новыми значениями без первого столбца для каждой категории
                df_ohe = new_df

            elif self.encoding_method == 'label':

                for col in self.cols_to_ohe_encode:
                    encoding_dict = self.label_encoding_dicts[col]
                    df_ohe[col] = df_ohe[col].map(encoding_dict)
                    ### Если на проде или на тестовой выборке мы получили новые категории,
                    # то заполняем неизвестные категории значением -1
                    if df_ohe[col].isna().sum() != 0:
                        df_ohe[col] = df_ohe[col].fillna(-1)

            else:
                raise ValueError("Unsupported encoding method. Please choose either 'one_hot' or 'label'.")
        
            df_ohe = df_ohe[sorted(df_ohe.columns)]



        if len(self.cols_to_mean_target_encode) != 0:

            if self.mte_strategy != "multiclass":
                for col in self.cols_to_mean_target_encode:
                    df_mte[col] = df_mte[col].map(self.dict_of_means[col])
                    df_mte[col] = self.fillnans(df_mte, col, self.dict_of_means[col])
            else:
                new_columns = {}
                for col in self.cols_to_mean_target_encode:
                    for target in self.mean_target_dict.keys():
                        new_col_name = f"{col}_{target}"
                        new_columns[new_col_name] = X[col].map(self.mean_target_dict[target][col])
                        new_columns[new_col_name] = self.fillnans(new_columns, new_col_name, self.mean_target_dict[target][col])

                    ### Удаляем один закодированный столбец по категории для того чтобы избежать мультиколлинеарности
                    del new_columns[f"{col}_{target}"]

                df_mte = df_mte.join(pd.DataFrame(new_columns))
                df_mte.drop(columns=self.cols_to_mean_target_encode, inplace=True)
        

        return pd.concat((df_orig, df_ohe, df_mte), axis=1)


    def fillnans(self, X, col, dict_of_means):

        ### Если на проде или на тестовой выборке мы получили новые категории, которые не были в обучении
        ### то заполняем эти данные по общему среднему всех полученных значений для того чтобы избавится от NaN-ов

        if X[col].isna().sum() != 0:
            if self.mte_fillna_strategy == "none":
                X[col] = X[col].fillna(0)
            elif self.mte_fillna_strategy == 'mean':
                mean_value = dict_of_means.mean()
                X[col] = X[col].fillna(mean_value)
            elif self.mte_fillna_strategy == 'mode':
                mode_value = X[col].mode()[0]
                X[col] = X[col].fillna(mode_value)
            else:
                raise ValueError("Unsupported fillna strategy. Please choose either 'none', 'mean' or 'mode'.")
        return X[col]


