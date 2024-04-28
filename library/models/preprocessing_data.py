import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(data, target_column):
    """
    Подготавливает данные для модели машинного обучения.

    Аргументы:
    data : pandas.DataFrame
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
    # Разделение данных на признаки и целевую переменную
    Y = data[target_column]
    X = data.drop([target_column], axis=1)

    # Преобразование численных признаков к одному масштабу
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2, random_state=42)

    return X_train, Y_train, X_test, Y_test


#! %%time 
#* # Разделим данные на обучающий и тестовый наборы
#~ X_train, Y_train, X_test, Y_test = prepare_data(data,'Y') 
#~ print("Данные разбиты")
#~ print("Train")
#~ display(features_train.head()) 
#~ print("Test") 
#~ display(features_test.head())

