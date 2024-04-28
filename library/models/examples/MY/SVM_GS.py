from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDClassifier 
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
 
pipe_SVM = Pipeline([ 
    ('scaler', StandardScaler()), 
    ('SVM', SGDClassifier(loss='hinge')) 
]) 
 
# Подбор лучшей модели с помощью GridSearchCV 
 
# Определение функции для вычисления PR AUC 
def pr_auc_score(y_true, y_pred_proba): 
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba) 
    pr_auc = auc(recall, precision) 
    return pr_auc 
 
# Создание объекта для применения метрики PR AUC в GridSearchCV 
pr_auc_scorer = make_scorer(pr_auc_score, needs_proba=False) 
 
param_grid = { 
    'SVM__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 
    'SVM__alpha': [0.0001, 0.001, 0.01, 0.1], 
    'SVM__penalty': ['l2', 'l1', 'elasticnet'], 
    'SVM__max_iter': [1000, 2000, 3000], 
} 
 
grid_search = GridSearchCV(pipe_SVM, 
                           param_grid,  
                           cv = 5,  
                           scoring=pr_auc_scorer) 
 
# Подгонка модели на обучающих данных 
grid_search.fit(X_train, y_train) 
 
# Вывод лучших параметров и оценки 
print("Лучшие параметры:", grid_search.best_params_) 
print("Лучшая оценка:", grid_search.best_score_)

# Назначаем лучшую модель 
best_pipe_SVM = grid_search.best_estimator_

#************************************************************
#~ KFold / регрессия


from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

# Создаем объект KFold для разбиения данных на фолды
splitter = KFold(n_splits=5, shuffle=True, random_state=33)

# Определение пайплайна 
SVM_pipe = Pipeline([
    ('scaler', StandardScaler()),  # Применяем стандартизацию признаков
    ('SVM', SVC())  # Модель SVM
]) 

# Замеряем метрики на кросс-валидации
SVM_cv = cross_validate(
    SVM_pipe,
    X,  # Признаки
    Y,  # Целевая переменная
    cv=splitter,  # Используем объект KFold для разбиения
    scoring='accuracy',  # Оценка качества по метрике accuracy
    return_train_score=True
)

# Выводим средние значения метрик на тренировочной и тестовой частях
print(f"Среднее accuracy модели SVM на трейне: {np.mean(SVM_cv['train_score']):.3f}")
print(f"Среднее accuracy модели SVM на тесте: {np.mean(SVM_cv['test_score']):.3f}")

# Если метрики удовлетворительные, можно обучить модель на всей выборке
SVM_pipe.fit(X, Y)