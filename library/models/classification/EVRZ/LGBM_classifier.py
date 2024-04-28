from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

def lgbm_classifier(random_state, X_train, Y_train, X_test, Y_test, array_out, c_w):
    """
    Создает классификатор LightGBM и вычисляет метрики качества.

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
    lgbm_results : pandas.DataFrame or tuple
        Результаты метрик качества или кортеж с максимальными метриками и важностью фичей.
    """
    print("LightGBMClassifier")
    lgbm_max_AUC = 0
    lgbm_AUC_depth = 0
    lgbm_max_F1 = 0
    lgbm_F1_depth = 0

    lgbm_data_metrix = []
    for depth in range(4, 21, 2):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'max_cat_threshold': 25,
            'min_data_in_leaf': 10,
            'num_threads': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': depth,
            'class_weight': c_w
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, Y_train)
        predicted_test = model.predict(X_test)

        # Вычисляем вектор вероятностей
        probabilities_test = model.predict_proba(X_test)
        probabilities_one_test = probabilities_test[:, 1]

        # расчет площади ‘auc_roc’
        auc_roc = round(roc_auc_score(Y_test, probabilities_one_test), 4)

        # расчет метрики f1
        f1 = round(f1_score(Y_test, predicted_test), 4)

        # Проверка Максимума
        if lgbm_max_AUC <= auc_roc:
            lgbm_max_AUC = auc_roc
            lgbm_AUC_depth = depth

        if lgbm_max_F1 <= f1:
            lgbm_max_F1 = f1
            lgbm_F1_depth = depth
        # пишем результаты в список
        lgbm_data_metrix.append([f1, auc_roc, depth])

    lgbm_data_metrix = pd.DataFrame(lgbm_data_metrix, columns=['f1', 'auc_roc', 'depth'])
    # print(lgbm_data_metrix)
    print('Максимум AUC =', lgbm_max_AUC, '| глубина дерева = ', lgbm_AUC_depth)
    print('Максимум F1 =', lgbm_max_F1, '| глубина дерева = ', lgbm_F1_depth)
    # Сохраняем важность фичей для лучшей модели

    best_model = LGBMClassifier(**params)
    best_model.fit(X_train, Y_train)
    feature_importance = pd.DataFrame(best_model.feature_importances_, index=X_train.columns)
    feature_importance.columns = ['importance']

    if array_out == 'No':
        return lgbm_data_metrix, feature_importance
    else:
        return (lgbm_max_AUC, lgbm_AUC_depth,
                lgbm_max_F1, lgbm_F1_depth), feature_importance
    

#! %%time  
#* #LGBM с подбором глубины дерева 
#~ LGBM_data_metrix, feature_importance = lgbm_classifier( 
#~    88811,
#~    X_train,
#~    Y_train,
#~    X_test,
#~    Y_test,
#~    'No',
#~    None
#~ )
