from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

def logistic_regression_classifier(random_state, X_train, Y_train, X_test, Y_test, array_out, class_weight=None):
    """
    Создает классификатор логистической регрессии и вычисляет метрики качества.

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
        Если 'No', возвращает DataFrame с метриками качества для различных значений C.
        Если 'Yes', возвращает максимальное значение AUC, соответствующее C,
        максимальное значение F1 и соответствующее C.
    class_weight : dict or 'balanced', default=None
        Веса классов для учета дисбаланса классов.

    Возвращает:
    Если array_out == 'No', возвращает DataFrame с метриками качества для различных значений C. 
    Если array_out == 'Yes', возвращает кортеж с максимальным значением AUC, соответствующим C, максимальным значением F1 и соответствующим C.
    """
    print("Logistic Regression Classifier")
    max_AUC = 0
    max_AUC_C = 0
    max_F1 = 0
    max_F1_C = 0

    data_metrics = []
    for i in range(-3, 4):
        C_value = 10 ** i
        model = LogisticRegression(random_state=random_state, solver='liblinear', class_weight=class_weight, C=C_value)
        model.fit(X_train, Y_train)
        predicted_test = model.predict(X_test)

        # Вычисление вектора вероятностей
        probabilities_test = model.predict_proba(X_test)
        probabilities_one_test = probabilities_test[:, 1]

        # Расчет AUC-ROC
        auc_roc = round(roc_auc_score(Y_test, probabilities_one_test), 4)

        # Расчет F1-меры
        f1 = round(f1_score(Y_test, predicted_test), 4)

        # Поиск максимальных значений метрик
        if max_AUC <= auc_roc:
            max_AUC = auc_roc
            max_AUC_C = C_value

        if max_F1 <= f1:
            max_F1 = f1
            max_F1_C = C_value

        data_metrics.append([f1, auc_roc, C_value])

    data_metrics = pd.DataFrame(data_metrics, columns=['f1', 'auc_roc', 'C_value'])

    print('Максимальное значение AUC =', max_AUC, '| С, соответствующий максимальному AUC =', max_AUC_C)
    print('Максимальное значение F1 =', max_F1, '| С, соответствующий максимальной F1 =', max_F1_C)

    if array_out == 'No':
        return data_metrics
    else:
        return max_AUC, max_AUC_C, max_F1, max_F1_C
    

#! %%time  
#* #лог ререссия с подбором С 
#~ LR_data_metrix = log_reg_clf( 
#~    88811,
#~    X_train,
#~    Y_train,
#~    X_test,
#~    Y_test,
#~    'No',
#~    None
#~ )


"""
Значения параметра class_weight могут быть заданы следующими способами:

1) None: Каждому классу присваивается одинаковый вес, то есть все классы считаются равнозначными.

2) 'balanced': Веса классов автоматически вычисляются обратно пропорционально их частоте в обучающем
наборе данных. Это означает, что классы, представленные реже, получат больший вес, а классы,
представленные чаще, получат меньший вес.

3) {class_label: weight}: Пользователь может явно указать вес для каждого класса. Это полезно,
когда есть особые соображения относительно значимости различных классов. Например, если
класс 0 важнее класса 1, можно назначить больший вес классу 0 и меньший вес классу 1.
"""