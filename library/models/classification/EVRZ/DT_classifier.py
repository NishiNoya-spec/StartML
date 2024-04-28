from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

def decision_tree_classifier(random_state, X_train, Y_train, X_test, Y_test, array_out, c_w):
    """
    Создает классификатор решающего дерева и вычисляет метрики качества.

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
    decision_tree_results : pandas.DataFrame or tuple
        Результаты метрик качества или кортеж с максимальными метриками.
    """
    print("DecisionTreeClassifier")
    DT_max_AUC = 0
    DT_AUC_depth = 0
    DT_max_F1 = 0
    DT_F1_depth = 0

    DT_data_metrics = []
    for depth in range(4, 21, 2):
        model = DecisionTreeClassifier(max_depth=depth, random_state=random_state, class_weight=c_w)
        model.fit(X_train, Y_train)
        predicted_test = model.predict(X_test)

        # Вычисляем вектор вероятностей
        probabilities_test = model.predict_proba(X_test)
        probabilities_one_test = probabilities_test[:, 1]

        # Расчет площади под ROC-кривой
        auc_roc = round(roc_auc_score(Y_test, probabilities_one_test), 4)

        # Расчет F1-меры
        f1 = round(f1_score(Y_test, predicted_test), 4)

        # Проверка максимумов
        if DT_max_AUC <= auc_roc:
            DT_max_AUC = auc_roc
            DT_AUC_depth = depth

        if DT_max_F1 <= f1:
            DT_max_F1 = f1
            DT_F1_depth = depth

        # Записываем результаты в список
        DT_data_metrics.append([f1, auc_roc, depth])

    DT_data_metrics = pd.DataFrame(DT_data_metrics, columns=['f1', 'auc_roc', 'depth'])

    print('Максимум AUC =', DT_max_AUC, '| Глубина дерева = ', DT_AUC_depth)
    print('Максимум F1 =', DT_max_F1, '| Глубина дерева = ', DT_F1_depth)

    if array_out == 'No':
        return DT_data_metrics
    else:
        return DT_max_AUC, DT_AUC_depth, DT_max_F1, DT_F1_depth
    

#! %%time  
#* #дерево решений с подбором глубины 
#~ DT_data_metrix = decision_tree_classifier( 
#~    88811,
#~    X_train,
#~    Y_train,
#~    X_test,
#~    Y_test,
#~    'No',
#~    None
#~ )
