from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score

def dummy_make(strategy, random_state, X_train, Y_train, X_test, Y_test):
    """
    Создает "фиктивный" классификатор и вычисляет метрики качества.

    Аргументы:
    strategy : str
        Стратегия классификации.
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

    Возвращает:
    dum_max_AUC : float
        Значение максимальной площади под ROC-кривой.
    dum_max_F1 : float
        Значение максимальной F1-меры.
    """
    print("DummyClassifier")
    # Создание и обучение классификатора
    dummy_clf = DummyClassifier(strategy=strategy, random_state=random_state)
    dummy_clf.fit(X_train, Y_train)
    predicted_test = dummy_clf.predict(X_test)

    # Вычисление вектора вероятностей
    probabilities_test = dummy_clf.predict_proba(X_test)
    probabilities_one_test = probabilities_test[:, 1]

    # Расчет площади под ROC-кривой
    auc_roc = round(roc_auc_score(Y_test, probabilities_one_test), 4)
    dum_max_AUC = auc_roc

    # Расчет F1-меры
    f1 = round(f1_score(Y_test, predicted_test), 4)
    dum_max_F1 = f1

    print('Максимум AUC =', dum_max_AUC)
    print('Максимум F1 =', dum_max_F1)

    return dum_max_AUC, dum_max_F1


"""
Фиктивный классификатор (Dummy Classifier) - это базовый алгоритм классификации,
который используется в качестве базовой линии для сравнения с другими классификаторами.
Он просто прогнозирует метки классов на основе простых правил или случайного выбора,
без учета входных данных. Этот классификатор используется для оценки производительности других,
более сложных моделей.

Фиктивный классификатор имеет несколько стратегий для генерации прогнозов, таких как:

"uniform": просто случайно выбирает метки классов с равными вероятностями для всех классов.
"stratified": генерирует прогнозы, сохраняя ту же долю каждого класса, как в обучающем наборе данных.
"most_frequent": просто предсказывает наиболее часто встречающийся класс в обучающем наборе данных.
"prior": прогнозирует классы в соответствии с априорными вероятностями классов в обучающем наборе данных.

Фиктивный классификатор обычно используется для оценки, насколько хорошо модель может разделить классы и
насколько ее результаты лучше, чем простые случайные прогнозы.
"""


#! %%time  
#* #фиктивный классификатор 
#~ DUM_max_AUC, DUM_max_F1 = dummy_make( 
#~    'most_frequent',
#~    88811,
#~    X_train,
#~    Y_train,
#~    X_test,
#~    Y_test,
#~    'No'
#~ )
