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