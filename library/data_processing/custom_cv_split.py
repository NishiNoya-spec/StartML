"""
Кросс-валидация исключает вероятность выбора необъективного дата-сета для обучения/валидации
(X_train, X_test, Y_train, Y_test) - путем разделения всей выборки на несколько частей для обучения
и валидации нескольких моделей - для последующего нахождения средней ошибки на всей кросс-валидации - 
называется обобщающей способностью модели.

- Почему модели могут обладать плохой обобщающей способностью:
1) Модель чрезмерно сложная или чрезмерно простая;
2) Неправильная спецификация задачи
3) В какой-то из выборок много выбросов
4) Данные засплитились сегментированно


Если обучаем модель на кросс-валидации, то важно помнить, что кол-во фолдов определяет кол-во полученных моделей,
то есть кол-во векторов весов.
Из этого так же следуюет, что все метрики должны усреднятся по всем полученным моделям от фолдов.

Замеряем K средних ошибок на валидации.
Если их среднее / их распределение нас устраивает, то строим финальную модель.
- Либо обучаем последний раз на всем train-е;
- Либо усредняем прогнозы / веса B (W) по всем фолдам;
- Либо берем лучшую на валидации

"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_data_by_time(X, y, time_column, split_date):
    """
    Отделяет валидационные данные от тренировочных по времени.

    Параметры:
    - X: DataFrame с признаками.
    - y: Series с целевой переменной.
    - time_column: имя столбца, содержащего информацию о времени.
    - split_date: дата, по которую происходит разделение.

    Возвращает:
    - X_train: DataFrame с тренировочными признаками.
    - X_val: DataFrame с валидационными признаками.
    - y_train: Series с тренировочными метками.
    - y_val: Series с валидационными метками.
    """
    X_train = X[X[time_column] < split_date]
    X_val = X[X[time_column] >= split_date]
    y_train = y[y.index.isin(X_train.index)]
    y_val = y[y.index.isin(X_val.index)]
    return X_train, X_val, y_train, y_val

###* Отделим валидацию от теста по времени

X_test, X_train = X[X.date >= '2017-06-01'], X[X.date < '2017-06-01']
y_test, y_train = y[y.index.isin(X_test.index)], y[y.index.isin(X_train.index)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from sklearn.model_selection import train_test_split, cross_validate

def custom_cv_split(X, y, test_size=0.25, random_state=None):
    """
    Функция для пользовательского разделения данных на обучающий и тестовый наборы, 
    возвращающая индексы для кросс-валидации.

    Параметры:
    - X: массив или DataFrame, содержащий признаки.
    - y: массив или Series, содержащий метки классов.
    - test_size: float, размер тестового набора (по умолчанию 0.25).
    - random_state: int, состояние генератора случайных чисел для воспроизводимости результатов.

    Возвращает:
    - custom_cv: список кортежей, содержащих индексы для кросс-валидации.
    - X_train: DataFrame, содержащий признаки обучающего набора.
    - X_test: DataFrame, содержащий признаки тестового набора.
    - y_train: Series, содержащий метки классов обучающего набора.
    - y_test: Series, содержащий метки классов тестового набора.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_indices = X_train.index.tolist()
    test_indices = X_test.index.tolist()
    custom_cv = [(train_indices, test_indices)]
    return custom_cv, X_train, X_test, y_train, y_test


"""

Отделить обучение, валидацию и тест, произвести подбор лучшей модели с помощью `K-Fold` на валидации,
финально обучить выбранную модель на всей валидации и замерить качество на заранее отложенном финальном тесте!

"""

# ==============================================================>>

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Разделение данных на обучающую + валидационную и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Определение пайплайна

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('LR', LogisticRegression(penalty='l2'))
])

# 3. Определение стратегии кросс-валидации (фолдов)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Подбор лучшей модели с помощью GridSearchCV

alphas = np.arange(0, 100, 0.1)

param_grid ={
    'LR__C': alphas
}

grid_search = GridSearchCV(pipe,
                           param_grid, 
                           cv = kfold.get_n_splits, 
                           scorring='accuracy')

# Подгонка модели на обучающих данных
grid_search.fit(X_train, y_train)

# Вывод лучших параметров и оценки
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая оценка:", grid_search.best_score_)

# 5. Выбор лучшей модели и обучение ее на всей обучающей выборки

# Теперь либо берем лучшую модель из GridSearchCV

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Либо назначаем лучшие гиперапараметры для pipe вручную

pipe.set_params(LR__C=grid_search.best_params_['LR__C'])
pipe.fit(X_train, y_train)

# 4. Проверка на тестовой выборке

test_score = pipe.score(X_test, y_test)
print(f"Accuracy на тестовом наборе: {test_score.round(3)}")


# ==============================================================>>

### Обучение без GridSearch с кастомным cross-validate (на примере LogisticRegression)

# 1. Определение пайплайна

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('LR', LogisticRegression(penalty='l2'))
])

# 2. Определение фолдов 

# Разделение данных на обучающий и тестовый наборы
custom_cv, X_train, X_test, y_train, y_test = custom_cv_split(X, y, test_size=0.25, random_state=1)

# 3. Подбор лучшей модели 

# 3.1 Функция cross_validate не возвращает прогнозы моделей на фолдах,
# она используется для оценки производительности модели на кросс-валидации. 

from sklearn.model_selection import cross_validate

cv_result_pipe = cross_validate(pipe,
                                X,
                                y,
                                scoring='accuracy',
                                cv=custom_cv,
                                return_train_score=True)

print(f"Accuracy на обучении (кросс-валидация): {np.mean(cv_result_pipe['train_score']).round(3)}")
print(f"Accuracy на валидации (кросс-валидация): {np.mean(cv_result_pipe['test_score']).round(3)}")

# 3.2 Обучение модели с использованием кросс-валидации

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# Получаем прогнозы на всех фолдах
y_pred_list = cross_val_predict(pipe, X, y, cv=custom_cv)

# Усредняем прогнозы
average_y_pred = np.mean(y_pred_list, axis=0)

# Оцениваем качество усредненных прогнозов
accuracy = accuracy_score(y, average_y_pred)
print("Accuracy на усредненных прогнозах:", accuracy)

# Обучение выбранной модели на всем обучающем наборе и оценка на тестовом наборе
pipe.fit(X_train, y_train)

# 4. Проверка на тестовой выборке

test_score = pipe.score(X_test, y_test)
print(f"Accuracy на тестовом наборе: {test_score.round(3)}")
