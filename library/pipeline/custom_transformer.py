from sklearn.base import BaseEstimator, TransformerMixin

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Преобразует данные, применяя пользовательскую функцию к выбранным столбцам.

    Параметры:
        function : функция для применения к данным
        columns : список столбцов, к которым применяется функция
    """

    def __init__(self, function, columns):
        self.function = function
        self.columns = columns

    def fit(self, X, y=None):
        """
        Ничего не делает, так как для простых преобразований данных не требуется обучение модели.
        """
        return self

    def transform(self, X, y=None):
        """
        Применяет функцию к выбранным столбцам данных.

        Параметры:
            X : pandas DataFrame
                Входные данные

        Возвращает:
            pandas DataFrame
                Преобразованные данные
        """
        X_ = X.copy()
        X_[self.columns] = self.function(X_[self.columns])
        return X_


#********************************

#^ Пример использования 

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, first_col, second_col, function):
        self.first_col = first_col
        self.second_col = second_col
        self.function = function
        print("Инициализировали класс!")
        
    def fit(self, X, y=None):
        print("Зафитили датасет!")
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        X_['new_feature'] = self.function(X_[self.first_col], X_[self.second_col])
        X_ = X_.drop([self.first_col, self.second_col], axis=1)
        
        print("Трансформировали датасет!")
        return X_

from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("custom_transformer",
    CustomFunctionTransformer(
        "x1",
        "x2",
        lambda x, y: 2*x**3 - 2*x**2 - x - y
    )),
    ("decision_tree", DecisionTreeClassifier(max_depth=2))
])

pipe.fit(X, y)

#! ********************************

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import itertools

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 object_columns=[],
                 target_name='y',
                 encoding_threshold=10,
                 fillna_strategy='mean',
                 encoding_method='one_hot'):
        
        self.object_columns = object_columns
        self.target_name = target_name
        self.encoding_threshold = encoding_threshold
        self.fillna_strategy = fillna_strategy
        self.encoding_method = encoding_method
        
                
    def fit(self,
            X,
            y):
        
        X_fit = X.copy()
        y_fit = y.copy()
        
        self.numeric_columns = [x for x in X_fit.columns if x not in self.object_columns]
        
        X_with_target = pd.concat((X_fit, y_fit), axis=1)
        
        ### Сгенерим колонки к которым применим One-Hot-Encoding
        self.cols_for_ohe = [col for col in self.object_columns
                             if 
                             X_with_target[col].nunique() <= self.encoding_threshold]
        
        ### Запомним все ohe колонки и их названия!
        self.ohe_names = {col : sorted([f"{col}_{value}" for value in X_with_target[col].unique()]) for col in self.cols_for_ohe}
        
        
        ### Сгенерим колонки к которым применим Mean-Target-Encoding
        self.cols_for_mte = [col for col in self.object_columns
                             if X_with_target[col].nunique() > self.encoding_threshold]
        
        ### Посчитаем на валидации средние значения таргета
        self.dict_of_means = {col : X_with_target.groupby(col)[self.target_name].mean()
                        for col in self.cols_for_mte}
        
        return self
    
    
    def transform(self,
                  X,
                  y=None):
        
        X_ = X.copy()
        
        
        if self.encoding_method == 'one_hot':
            data_part = pd.get_dummies(X_[self.cols_for_ohe],
                                       prefix=self.cols_for_ohe)
            data_part.replace({False: 0, True: 1}, inplace=True)
        elif self.encoding_method == 'label':
            data_part = self._label_encoding(X_)
        else:
            raise ValueError("Unsupported encoding method. Please choose either 'one_hot' or 'label'.")

        data_part_cols = data_part.columns
        
        X_ = X_.drop(self.cols_for_ohe, axis=1)
        X_ = pd.concat((X_, data_part), axis=1)
        
    
        for col in self.cols_for_mte:
            X_[col] = X_[col].map(self.dict_of_means[col])
                
            if self.fillna_strategy == 'mean':
                mean_value = self.dict_of_means[col].values.mean()
                X_[col] = X_[col].fillna(mean_value)
            elif self.fillna_strategy == 'mode':
                mode_value = X_[col].mode()[0]
                X_[col] = X_[col].fillna(mode_value)
            else:
                raise ValueError("Unsupported fillna strategy. Please choose either 'mean' or 'mode'.")
                
            
            
        all_ohe = list(itertools.chain(*list(self.ohe_names.values())))
        
        missing_columns = [x 
                           for x in all_ohe
                           if x not in X_.columns
                           and
                           x not in self.numeric_columns]

        extra_columns = [x
                         for x in data_part_cols
                         if x not in all_ohe]
        
        ### Новые категории необходимо убрать
        X_ = X_.drop(extra_columns, axis=1)
    
        ### Отсутствующие категории (бинарные колонки)
        ### необходимо добавить: заполним их просто нулями
        
        if len(missing_columns) != 0:

            zeros = np.zeros((X_.shape[0], len(missing_columns)))
            zeros = pd.DataFrame(zeros,
                                 columns=missing_columns,
                                 index=X_.index)

            X_ = pd.concat((X_, zeros), axis=1)
            
        return X_[sorted(X_.columns)]
    
    def _label_encoding(self, X):
        label_encoders = {}
        for col in self.cols_for_ohe:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        return X
