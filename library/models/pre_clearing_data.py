import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_model_clearing(data_x, target_column):
    """
    Очищает данные от выбросов с использованием изоляционного леса и подготавливает их для модели.

    Аргументы:
    data_x : pandas.DataFrame
        Исходные данные.
    target_column : str
        Название целевой переменной.

    Возвращает:
    X_train : pandas.DataFrame
        Признаки для обучающей выборки.
    Y_train : pandas.Series
        Целевая переменная для обучающей выборки.
    X_test : pandas.DataFrame
        Признаки для тестовой выборки.
    Y_test : pandas.Series
        Целевая переменная для тестовой выборки.
    """
    data = data_x.copy()

    # Обучение изолирующего леса для обнаружения выбросов
    isolation_forest = IsolationForest(n_estimators=300, random_state=88111)
    isolation_forest.fit(data)
    
    # Определение выбросов
    estimator = isolation_forest.fit_predict(data)
    outliers = estimator[estimator == -1]  # Выбросы имеют значение -1
    
    print("Количество аномалий: ", len(outliers))
    print("Всего данных: ", len(estimator))
    
    # Добавление маркеров выбросов к данным
    data['estimator'] = estimator
    
    # Создание выборки без выбросов
    data = data[data['estimator'] != -1].drop(['estimator'], axis=1).reindex()

    # Разделение данных на признаки и целевую переменную
    Y = data[target_column]
    X = data.drop([target_column], axis=1)
    
    # Масштабирование численных признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        shuffle=True, test_size=0.2,
                                                        random_state=88811)

    return X_train, Y_train, X_test, Y_test


"""
В методе изоляционного леса каждая точка данных оценивается на основе того,
насколько быстро модель смогла изолировать ее от остальных данных.
Если точка изолирована раньше и быстрее, то ей присваивается более высокая оценка,
близкая к 1. Если точка сложно изолируется и находится вне густой области данных,
ей присваивается более низкая оценка, близкая к -1. Точки с оценкой близкой к -1
считаются выбросами, тогда как точки с оценкой близкой к 1 считаются "нормальными" данными.
"""


