from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, precision_recall_curve, auc 
from sklearn.model_selection import train_test_split 
 
# Разделим данные на обучающий и тестовый наборы

X = df.drop(['Y'], axis=1) 
Y = df['Y'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#************************************************************
#~ GridSearchCV / классификация

# Определение пайплайна 
 
pipe_LR = Pipeline([ 
    ('scaler', StandardScaler()), 
    ('LR', LogisticRegression(penalty='l2')) 
]) 
 
# Подбор лучшей модели с помощью GridSearchCV 
 
# Определение функции для вычисления PR AUC 
def pr_auc_score(y_true, y_pred_proba): 
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba) 
    pr_auc = auc(recall, precision) 
    return pr_auc 
 
# Создание объекта для применения метрики PR AUC в GridSearchCV 
pr_auc_scorer = make_scorer(pr_auc_score, needs_proba=True) 
 
param_grid = { 
    'LR__penalty': ['l1', 'l2', 'none'],  # тип регуляризации 
    'LR__C': [0.001, 0.01, 0.1, 1, 10, 100],  # обратный коэффициент регуляризации 
    'LR__solver': ['liblinear', 'saga'],  # алгоритм оптимизации 
    'LR__max_iter': [100, 200, 300, 500, 1000],  # максимальное число итераций 
} 
 
grid_search = GridSearchCV(pipe_LR, 
                           param_grid,  
                           cv = 5,  
                           scoring=pr_auc_scorer) 
 
# Подгонка модели на обучающих данных 
grid_search.fit(X_train, y_train) 
 
# Вывод лучших параметров и оценки 
print("Лучшие параметры:", grid_search.best_params_) 
print("Лучшая оценка:", grid_search.best_score_)

# Назначаем лучшую модель 
best_pipe_LR = grid_search.best_estimator_


#************************************************************
#~ KFold / классификация

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Создаем объект KFold для разбиения данных на фолды
splitter = KFold(n_splits=5, shuffle=True, random_state=33)

# Определение пайплайна 
LR_pipe = Pipeline([
    ('scaler', StandardScaler()),  # Применяем стандартизацию признаков
    ('LR', LogisticRegression())  # Модель логистической регрессии
]) 

# Замеряем метрики на кросс-валидации
LR_cv = cross_validate(
    LR_pipe,
    X,  # Признаки
    Y,  # Целевая переменная
    cv=splitter,  # Используем объект KFold для разбиения
    scoring='accuracy',  # Оценка качества по метрике accuracy
    return_train_score=True
)

# Выводим средние значения метрик на тренировочной и тестовой частях
print(f"Среднее accuracy модели LR на трейне: {np.mean(LR_cv['train_score']):.3f}")
print(f"Среднее accuracy модели LR на тесте: {np.mean(LR_cv['test_score']):.3f}")

# Если метрики удовлетворительные, можно обучить модель на всей выборке
LR_pipe.fit(X, Y)

