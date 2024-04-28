from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, precision_recall_curve, auc, accuracy_score 
from sklearn.model_selection import train_test_split 
import numpy as np

# Разделим данные на обучающий и тестовый наборы

X = df.drop(['Y'], axis=1) 
Y = df['Y'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#& ------------------------------------>
# Перевзвешивание через Гаусовское ядро

def kernel(distances, h):

    const = 1 / (np.sqrt(2 * np.pi))
    power = (-1/2) * ((distances)**2) / h**2

    return const * np.exp(power)

"""
Возможные значения параметра KNN__weights:

1) 'uniform': Все соседи имеют одинаковый вес. Это означает, что все соседи входят
в равной степени в решение о классификации или регрессии для данного наблюдения.

2) 'distance': Вес каждого соседа обратно пропорционален расстоянию до данного
наблюдения. Таким образом, ближайшие соседи будут иметь больший вес, чем более удаленные.

3) 'kernel': Веса соседей вычисляются с использованием ядерной функции, которая зависит
от расстояния между наблюдением и его соседями. Такие ядерные функции могут быть различными,
например, гауссовским ядром.

"""

#& ------------------------------------>

#************************************************************
#~ GridSearchCV / классификация

# Определение пайплайна
pipe_KNN = Pipeline([
    ('scaler', StandardScaler()),  # Стандартизация признаков
    ('KNN', KNeighborsRegressor())  # Модель KNN
])

# Определение параметров для GridSearchCV
param_grid = {
    'KNN__n_neighbors': [1, 2, 3],  # Количество соседей
    'KNN__weights': ['uniform', 'distance'] + [lambda x, h=h: kernel(x, h) for h in [0.01, 0.05, 0.1, 0.2]],  # Веса соседей + Гауссовское ядро
    'KNN__p': [0.5, 1, 2],  # Параметр метрики расстояния
}
 
# Создание объекта GridSearchCV
grid_search = GridSearchCV(pipe_KNN,
                           param_grid,
                           cv=5,  # Количество фолдов в кросс-валидации
                           scoring='neg_root_mean_squared_error')

# Подгонка модели на обучающих данных
grid_search.fit(X_train, y_train)

# Вывод лучших параметров и оценки
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая оценка:", grid_search.best_score_)

# Получение лучшей модели
best_pipe_KNN = grid_search.best_estimator_

# Оценка качества на тестовом наборе данных
RMSE_train = (np.mean(best_pipe_KNN.predict(X_train) - y_train)**2)**0.5
RMSE_test = (np.mean(best_pipe_KNN.predict(X_test) - y_test)**2)**0.5
print(f"RMSE KNN на трейне: {RMSE_train:.3f}")
print(f"RMSE KNN на тесте: {RMSE_test:.3f}")

#^------------------------->

# Оценка качества на тестовом наборе данных
y_pred = best_pipe_KNN.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy на тестовом наборе данных:", accuracy)

#^------------------------->

#************************************************************
#~ KFold / регрессия

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

splitter = KFold(n_splits=5, shuffle=True, random_state=33)

# Определение пайплайна 
 
KNN_pipe = Pipeline([ 
    ('scaler', StandardScaler()), 
    ('KNN', KNeighborsRegressor(n_neighbors=3, weights=lambda x: kernel(x, h))) 
]) 

# Замеряем метрики на кросс-валидации

KNN_cv = cross_validate(
    KNN_pipe,
    X, 
    Y, 
    cv=splitter,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

display(f"Среднее MSE линейной модели на трейне: {-np.mean(KNN_cv['train_score']):.3f}")
display(f"Среднее MSE линейной модели на тесте: {-np.mean(KNN_cv['test_score']):.3f}")

# Если метрики удовлетворительны, то обучаем на всей выборке 

KNN_pipe.fit(X, Y)



"""
Гиперпараметр \( h \) в Гауссовском ядре определяет ширину окна, которое определяет, какие соседи будут учитываться при прогнозировании для нового примера.
Если значение \( h \) маленькое, то окно будет узким, и будут учитываться только ближайшие соседи, которые попадают в область ядра. Это может привести к более чувствительной модели, которая может быть более склонна к переобучению.
Если значение \( h \) большое, то окно будет широким, и большее количество соседей попадет в область ядра. Это может привести к усреднению весов объектов, которые находятся близко к рассматриваемому объекту, и к упрощению модели. В этом случае модель может быть менее чувствительной к выбросам или шуму в данных, но также может потерять способность различать различные образцы данных.
Поэтому важно правильно подобрать значение гиперпараметра \( h \) в Гауссовском ядре, чтобы достичь баланса между учетом плотности данных и избежанием переобучения или недообучения модели.
"""