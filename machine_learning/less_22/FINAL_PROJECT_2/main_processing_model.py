import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import matplotlib.pyplot as plt # подключаем графики

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from catboost import CatBoostClassifier

from sklearn.utils.class_weight import compute_class_weight

import os
import pickle


def batch_load_sql(query: str) -> pd.DataFrame:

    ### 7. Читаем записанный DataFrame из базы данных -->>

    # Функция для чтения признаков из базы данных батчами

    CHUNKSIZE = 200000
    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()

    return pd.concat(chunks, ignore_index=True)

def load_features(table_name) -> pd.DataFrame:
    ### 8. Читаем DataFrame из базы данных -->>
    query = f"SELECT * FROM {table_name}"
    return batch_load_sql(query)

def load_to_sql(table_name, data):

    ### 6. Записываем DataFrame в базу данных -->>

    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")
    data.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=10000)

# Утилиты

def train_test_split_sorted(data, train_size=0.8):
    ### Отсортируем данные по дате
    print("Сортировка данных по дате: ")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data.reset_index(drop=True, inplace=True)
    ### Делим выборку 80 на 20
    split_index = int(len(data) * train_size)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    train.drop(["timestamp"], axis=1, inplace=True)
    test.drop(["timestamp"], axis=1, inplace=True)
    print("Предварительная выборка на трейн: ")
    print(train)
    print("Предварительная выборка на тест: ")
    print(test)

    return train, test


### Функция для построения гистограммы важности признаков
def plot_feature_importances(feature_importance, model_name, target_type):
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    plt.figure(figsize=(20, 16))
    plt.barh(feature_importance.index, feature_importance.importance, height=0.7)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'{model_name} - {target_type} - Feature Importance', fontsize=16)
    plt.show()


### Строит матрицу путаниц для оценки качества классификации
def ax_plot_confusion_matrix(ax, y_true, y_pred, labels=None, title="Confusion Matrix"):
    # Вычисляем матрицу путаниц
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Создаем объект для отображения матрицы путаниц
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # Отображаем матрицу путаниц на графике
    cmp.plot(ax=ax)

    ax.set_title(title)


def lgbm_clf(x_random_state, X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed, c_w,
             metric='AUC-ROC'):
    """
    Классификатор LGBMClassifier с подбором гиперпараметров по метрике AUC-ROC или F1-Score.
    """
    print("LightGBMClassifier on Transformed Data")

    # Начальные значения для поиска наилучших гиперпараметров
    max_metric = 0
    best_depth = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=x_random_state)
    metrics_data = []

    # Определяем метрику
    if metric == 'AUC-ROC':
        metric_func = roc_auc_score
        metric_name = 'AUC-ROC'
    elif metric == 'F1-Score':
        metric_func = f1_score
        metric_name = 'F1-Score'
    else:
        raise ValueError("Invalid metric. Choose either 'AUC-ROC' or 'F1-Score'.")

    # Перебор различных значений глубины дерева
    for depth in tqdm(range(4, 51, 2), desc="Depth Progress",
                      leave=True):  # Внешний прогресс-бар для перебора глубины дерева
        fold_metrics = []

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'max_cat_threshold': 25,
            'min_data_in_leaf': 10,
            'n_jobs': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': depth,
            'class_weight': c_w
        }

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_transformed, y_train_transformed)):
            X_fold_train, X_valid = X_train_transformed.iloc[train_idx], X_train_transformed.iloc[valid_idx]
            y_fold_train, y_valid = y_train_transformed.iloc[train_idx], y_train_transformed.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train.to_numpy().ravel())

            # Предсказания и вычисление метрики
            if metric == 'AUC-ROC':
                y_pred = model.predict_proba(X_valid)[:, 1]
            else:
                y_pred = model.predict(X_valid)

            metric_value = metric_func(y_valid, y_pred)
            fold_metrics.append(metric_value)

        avg_metric = np.mean(fold_metrics)
        metrics_data.append([avg_metric, depth])

        if max_metric < avg_metric:
            max_metric = avg_metric
            best_depth = depth

    # Сохранение данных о метриках
    metrics_data = pd.DataFrame(metrics_data, columns=[metric_name, 'Depth'])
    metrics_data["best_depth"] = best_depth
    metrics_data["max_train_metric"] = max_metric
    print(f'Maximum {metric_name} = {max_metric:.4f} | Best Depth = {best_depth}')

    # Обучение на всех тренировочных данных с лучшими гиперпараметрами
    best_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'max_cat_threshold': 25,
        'min_data_in_leaf': 10,
        'n_jobs': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': best_depth,
        'class_weight': c_w
    }

    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train_transformed, y_train_transformed.to_numpy().ravel())

    # Оценка на тестовых данных
    y_pred_test_auc_roc = best_model.predict_proba(X_test_transformed)[:, 1]
    y_pred_test_f1 = best_model.predict(X_test_transformed)

    metrics_data["max_test_auc_roc"] = roc_auc_score(y_test_transformed, y_pred_test_auc_roc)
    metrics_data["max_test_f1"] = f1_score(y_test_transformed, y_pred_test_f1)

    if metric == 'AUC-ROC':
        y_pred_test = best_model.predict_proba(X_test_transformed)[:, 1]
    else:
        y_pred_test = best_model.predict(X_test_transformed)

    test_metric = metric_func(y_test_transformed, y_pred_test)
    print(f'Test {metric_name}: {test_metric:.4f}')

    print(metrics_data)

    # Анализ важных фичей на основе важности признаков
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame(
        importances,
        index=X_train_transformed.columns,
        columns=['importance']
    ).sort_values(by='importance', ascending=False)

    # Построение графика важности признаков
    plot_feature_importances(
        feature_importance=feature_importance,
        model_name=f"LGBM {c_w}",
        target_type=metric_name
    )

    return best_model, metrics_data


def catboost_clf(x_random_state, X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed, c_w,
                 cat_features, metric='AUC-ROC'):
    """
    Классификатор CatBoostClassifier с подбором гиперпараметров по метрике AUC-ROC или F1-Score.
    """
    print("CatBoostClassifier on Transformed Data")

    # Рассчитываем веса классов, если указано 'balanced'
    if c_w == 'balanced':
        class_labels = np.unique(y_train_transformed)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train_transformed)
        c_w = dict(zip(class_labels, class_weights))

    # Начальные значения для поиска наилучших гиперпараметров
    max_metric = 0
    best_depth = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=x_random_state)
    metrics_data = []

    # Определяем метрику
    if metric == 'AUC-ROC':
        metric_func = roc_auc_score
        metric_name = 'AUC-ROC'
    elif metric == 'F1-Score':
        metric_func = f1_score
        metric_name = 'F1-Score'
    else:
        raise ValueError("Invalid metric. Choose either 'AUC-ROC' or 'F1-Score'.")

    # Перебор различных значений глубины дерева
    for depth in tqdm(range(4, 14), desc="Depth Progress",
                      leave=True):  # Внешний прогресс-бар для перебора глубины дерева
        fold_metrics = []
        params = {
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': depth,
            'eval_metric': 'AUC' if metric == 'AUC-ROC' else 'F1',
            'random_seed': x_random_state,
            'logging_level': 'Silent',
            'class_weights': c_w
        }
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_transformed, y_train_transformed)):
            X_fold_train, X_valid = X_train_transformed.iloc[train_idx], X_train_transformed.iloc[valid_idx]
            y_fold_train, y_valid = y_train_transformed.iloc[train_idx], y_train_transformed.iloc[valid_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_fold_train, y_fold_train.to_numpy().ravel(), cat_features=cat_features)

            # Предсказания и вычисление метрики
            if metric == 'AUC-ROC':
                y_pred = model.predict_proba(X_valid)[:, 1]
            else:
                y_pred = model.predict(X_valid)
            metric_value = metric_func(y_valid, y_pred)
            fold_metrics.append(metric_value)

        avg_metric = np.mean(fold_metrics)
        metrics_data.append([avg_metric, depth])
        if max_metric < avg_metric:
            max_metric = avg_metric
            best_depth = depth

    # Сохранение данных о метриках
    metrics_data = pd.DataFrame(metrics_data, columns=[metric_name, 'Depth'])
    metrics_data["best_depth"] = best_depth
    metrics_data["max_train_metric"] = max_metric
    print(f'Maximum {metric_name} = {max_metric:.4f} | Best Depth = {best_depth}')

    # Обучение на всех тренировочных данных с лучшими гиперпараметрами
    best_params = {
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': best_depth,
        'eval_metric': 'AUC' if metric == 'AUC-ROC' else 'F1',
        'random_seed': x_random_state,
        'logging_level': 'Silent',
        'class_weights': c_w
    }
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train_transformed, y_train_transformed.to_numpy().ravel(), cat_features=cat_features)

    # Оценка на тестовых данных
    y_pred_test_auc_roc = best_model.predict_proba(X_test_transformed)[:, 1]
    y_pred_test_f1 = best_model.predict(X_test_transformed)
    metrics_data["max_test_auc_roc"] = roc_auc_score(y_test_transformed, y_pred_test_auc_roc)
    metrics_data["max_test_f1"] = f1_score(y_test_transformed, y_pred_test_f1)

    if metric == 'AUC-ROC':
        y_pred_test = best_model.predict_proba(X_test_transformed)[:, 1]
    else:
        y_pred_test = best_model.predict(X_test_transformed)

    test_metric = metric_func(y_test_transformed, y_pred_test)
    print(f'Test {metric_name}: {test_metric:.4f}')
    print(metrics_data)

    # Анализ важных фичей на основе важности признаков
    importances = best_model.get_feature_importance()
    feature_importance = pd.DataFrame(
        importances,
        index=X_train_transformed.columns,
        columns=['importance']
    ).sort_values(by='importance', ascending=False)

    # Построение графика важности признаков
    plot_feature_importances(
        feature_importance=feature_importance,
        model_name=f"CatBoost {c_w}",
        target_type=metric_name
    )
    return best_model, metrics_data


def save_models_pkl(model, model_name, file_path):
    # Создаем директорию для модели
    model_dir = os.path.join(file_path, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Полный путь к файлу для сохранения модели
    model_file_path = os.path.join(model_dir, f"{model_name}.pkl")

    # Сохраняем модель с помощью pickle
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model {model_name} saved successfully to {model_file_path}.")


def save_model_cbm(model, model_name, file_path):
    """
    Сохраняет модель CatBoost в формате .cbm.

    Параметры:
    - model: CatBoostClassifier - модель CatBoost для сохранения.
    - model_name: str - имя модели для сохранения.
    - file_path: str - путь к директории, в которой будет сохранена модель.
    """
    # Создаем директорию для модели
    model_dir = os.path.join(file_path, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Полный путь к файлу для сохранения модели
    model_file_path = os.path.join(model_dir, f"{model_name}.cbm")

    # Сохраняем модель с помощью метода save_model
    model.save_model(model_file_path, format="cbm")

    print(f"Model {model_name} saved successfully to {model_file_path}.")


if __name__ == "__main__":


    path_data = "C:/python_projects/FinalProject_2/data/"
    path_models = "C:/python_projects/FinalProject_2/models/"

    """
    all_data = pd.read_csv(path_data + "all_features_with_targets.csv", sep=',')

    train_data, test_data = train_test_split_sorted(all_data, train_size=0.8)

    train_data.to_csv(path_data + "train_data.csv", sep=',', index=0)
    test_data.to_csv(path_data + "test_data.csv", sep=',', index=0)
    """

    """
    path_data = "C:/python_projects/FinalProject_2/data/"

    all_data = pd.read_csv(path_data + "all_features_with_targets.csv", sep=',')
    print(all_data)

    print(all_data.isna().sum())

    load_to_sql("danil_temnkhudov_features_lesson_22_with_target", all_data)
    """

    """
    path_data = "C:/python_projects/FinalProject_2/data/"

    print("Загрузка данных:")

    features = pd.read_csv(path_data + "all_user_features_data_new.csv", sep=',')

    print(f"features: {features}")

    target_data = pd.read_csv(path_data + "target_data.csv", sep=',')

    print(f"target_data: {target_data}")

    # Объединение таблиц по столбцам user_id и post_id
    # Используем метод merge и указываем, что нужно объединять по этим столбцам
    merged_data = pd.merge(features, target_data, on=['user_id', 'post_id'], how='inner')

    # Дропаем дупликаты
    merged_data = merged_data.drop_duplicates(subset=["user_id", "post_id"], keep="last")

    # Сохраняем результат в новый CSV-файл, если нужно
    merged_data.to_csv(path_data + 'all_features_with_targets.csv', sep=',', index=0)

    # Показать первые строки для проверки
    print(merged_data.head())
    print(merged_data.shape)
    """


    """
    path_data = "data/"

    # Создаем строку соединения с базой данных
    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")

    # Открываем соединение
    with engine.connect() as connection:
        # Выполняем запрос и получаем данные
        feed_data = pd.read_sql("SELECT * FROM public.feed_data", con=connection)

    feed_data.to_csv(path_data + "feed_data_all.csv", sep=',', index=0)
    feed_data = pd.read_csv(path_data + "feed_data.csv", sep=',')
    print(feed_data)

    """

    """
    # Создание target_data.csv
    path_data = "C:/python_projects/FinalProject_2/data/"
    chunksize = 50000
    feed_data = pd.read_csv(path_data + "feed_data_all.csv", sep=",")
    #feed_data.drop(["timestamp"], axis=1, inplace=True)

    ### Выделим признак, для того чтобы понимать лайкнул пост юзер или нет
    def create_like_counter(row):
        if row['action'] == 'like' or row['target'] == 1:
            return 1
        else:
            return 0

    feed_data["like_target"] = feed_data.apply(create_like_counter, axis=1)
    feed_data.drop(["action", "target"], axis=1, inplace=True)

    feed_data.to_csv(path_data + "target_data.csv", sep=',', index=0)
    target_data = pd.read_csv(path_data + "target_data.csv", sep=',')
    print(target_data)
    """


    """
    # Выгружаем из БД сформированные признаки с таргетом и timestamp-ом
    #table_name = 'danil_temnkhudov_features_lesson_22_with_target'
    #features_with_target = load_features(table_name=table_name)

    features_with_target = pd.read_csv(path_data + "all_features_with_targets.csv", sep=',')

    print(features_with_target)

    train_data, test_data = train_test_split_sorted(features_with_target, train_size=0.8)

    X_train = train_data.drop(["like_target"], axis=1)
    X_test = test_data.drop(["like_target"], axis=1)
    y_train = train_data["like_target"]
    y_test = test_data["like_target"]

    cat_features = [
        'country',
        'city',
        'topic',
        'weekday_cat'
    ]

    CatB_best_model, CatB_metrics_data = catboost_clf(
        x_random_state=42,
        X_train_transformed=X_train,
        y_train_transformed=y_train,
        X_test_transformed=X_test,
        y_test_transformed=y_test,
        c_w='balanced',  # или другое значение class_weight ('balanced' / None)
        cat_features=cat_features,
        metric='AUC-ROC'  # AUC-ROC / F1-Score
    )

    print(f"metrics: {CatB_metrics_data}")

    print(f"model: {CatB_best_model}")

    """

    ### 1. Загрузка всех трансформированных данных
    features_with_target = pd.read_csv(path_data + "all_features_with_targets.csv", sep=',')
    features_with_target.drop(["timestamp"], axis=1, inplace=True)
    print(features_with_target)

    X = features_with_target.drop(["like_target", "user_id", "post_id"], axis=1)
    y = features_with_target['like_target']

    ### Взвешивание
    class_labels = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=y
    )
    c_w = dict(zip(class_labels, class_weights))

    cat_features = [
        'country',
        'city',
        'topic',
        'weekday_cat'
    ]

    # Обучение на всех тренировочных данных с лучшими гиперпараметрами
    best_params = {
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': 10,
        'eval_metric': 'AUC', # 'F1'
        'random_seed': 42,
        'logging_level': 'Silent',
        'class_weights': c_w
    }
    model = CatBoostClassifier(**best_params)
    model.fit(X, y.to_numpy().ravel(), cat_features=cat_features)

    save_model_cbm(
        model,
        "CatBoost",
        path_models
    )

    print(f"model: {model}")

