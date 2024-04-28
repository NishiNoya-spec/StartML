from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

def random_forest_classifier(random_state, X_train, Y_train, X_test, Y_test, array_out, c_w):
    """
    Создает случайный лес классификатор и вычисляет метрики качества.

    Аргументы:
    random_state : int
        Значение для инициализации генератора случайных чисел.
    X_train : pandas.DataFrame
        Признаки обучающей выборки.
    Y_train : pandas.Series
        Целевая переменная обучающей выборки.
    X_test : pandas.DataFrame
        Признаки тестовой выборки.
    Y_test : pandas.Series
        Целевая переменная тестовой выборки.
    array_out : str
        Флаг для указания на вывод результатов в виде кортежа.
    c_w : str or dict
        Параметр для учета весов классов.

    Возвращает:
    random_forest_results : pandas.DataFrame or tuple
        Результаты метрик качества или кортеж с максимальными метриками.
    """
    print("RandomForestClassifier")
    # Метрики качества для перебора
    RF_max_AUC = 0
    RF_AUC_n_estimators = 0
    RF_AUC_depth = 0
    RF_max_F1 = 0
    RF_F1_n_estimators = 0
    RF_F1_depth = 0

    # Список значений метрик
    RF_data_metrix = []

    for depth in range(6, 11, 2):
        for estim in range(50, 301, 50):
            model = RandomForestClassifier(n_estimators=estim, max_depth=depth, random_state=random_state,
                                           class_weight=c_w)
            model.fit(X_train, Y_train)
            predicted_test = model.predict(X_test)

            # Вычисляем вектор вероятностей
            probabilities_test = model.predict_proba(X_test)
            probabilities_one_test = probabilities_test[:, 1]

            # Расчет площади ‘auc_roc’
            auc_roc = round(roc_auc_score(Y_test, probabilities_one_test), 4)

            # Расчет метрики F1
            f1 = round(f1_score(Y_test, predicted_test), 4)

            # Проверка максимумов
            if RF_max_AUC <= auc_roc:
                RF_max_AUC = auc_roc
                RF_AUC_n_estimators = estim
                RF_AUC_depth = depth

            if RF_max_F1 <= f1:
                RF_max_F1 = f1
                RF_F1_depth = depth
                RF_F1_n_estimators = estim

            # Записываем результаты в список
            RF_data_metrix.append([f1, auc_roc, depth, estim])

    RF_data_metrix = pd.DataFrame(RF_data_metrix, columns=['f1', 'auc_roc', 'depth', 'estim'])
    # print(RF_data_metrix)
    print('Максимум AUC =', RF_max_AUC, '| число деревьев = ', RF_AUC_n_estimators, '| глубина дерева = ', RF_AUC_depth)
    print('Максимум F1 =', RF_max_F1, '| число деревьев = ', RF_F1_n_estimators, '| глубина дерева = ', RF_F1_depth)

    if array_out == 'No':
        return RF_data_metrix
    else:
        return (RF_max_AUC, RF_AUC_n_estimators, RF_AUC_depth,
                RF_max_F1, RF_F1_n_estimators, RF_F1_depth)
    

#! %%time  
#* #случайный лес с подбором количества деревьев и глубины дерева 
#~ RF_data_metrix = random_forest_classifier( 
#~    88811,
#~    features_train,
#~    target_train,
#~    features_test,
#~    target_test,
#~    'No',
#~    None
#~ )


