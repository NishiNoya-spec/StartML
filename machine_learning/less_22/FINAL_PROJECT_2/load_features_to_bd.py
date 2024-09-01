import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from tqdm import tqdm
import gc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder as SklearnOneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_engine():
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    return create_engine(SQLALCHEMY_DATABASE_URL)


def execute_query(engine, query, chunksize=50000):
    with engine.connect() as connection:
        for chunk in pd.read_sql(query, connection, chunksize=chunksize):
            yield chunk


def load_table(engine, table_name, chunksize):
    data_chunks = []
    query = f"SELECT * FROM public.{table_name}"
    for chunk in tqdm(execute_query(engine, query, chunksize=chunksize), desc=f"Loading {table_name}..."):
        data_chunks.append(chunk)
        del chunk
        gc.collect()
    data = pd.concat(data_chunks)
    return data


def load_initial_tables(engine, chunksize=50000, feed_data_limit=50000):
    user_data = load_table(engine, "user_data", chunksize)
    post_text_df = load_table(engine, "post_text_df", chunksize)
    if feed_data_limit == None:
        feed_data = load_table(engine, f"feed_data", chunksize)
    else:
        feed_data = load_table(engine, f"feed_data limit {feed_data_limit}", chunksize)

    # Объединение таблиц
    data = feed_data.merge(user_data, on='user_id', how='inner')
    data = data.merge(post_text_df, on='post_id', how='inner')

    ### Отсортируем данные по дате
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data.reset_index(drop=True, inplace=True)

    return data


def prepare_data(data):
    ### 1. Отмечаем начальные признаки из БД -->>

    # Список категориальных признаков
    categorical_cols = ["country", "city", "os", "source", "topic"]
    # Список числовых признаков
    numeric_cols = []
    # Текстовые признаки
    text_cols = ['text']
    # Список столбцов, для которых не производим первичную обработку (кодирование, скалирование и тд.)
    passthrough_cols = ["gender", "age", "exp_group", "post_id"]

    ### 2. Сформируем таргетные переменные -->>

    # Создаем таргет like_target - признак того, что юзер лайкнул пост или нет
    data["like_target"] = data.apply(lambda row: 1 if row['action'] == 'like' or row['target'] == 1 else 0, axis=1)
    data.drop(["action", "target"], axis=1, inplace=True)
    target_name = ["like_target"]

    ### 3. Дропаем осатвшиеся ненужные признаки -->>

    # data.drop(["timestamp", "post_id", "text"], axis=1, inplace=True)
    data.set_index("user_id", inplace=True)

    ### 4. Разделим на X и y -->>

    X_data = data.drop("like_target", axis=1)
    y_data = data["like_target"]

    return data, X_data, y_data, categorical_cols, numeric_cols, text_cols, passthrough_cols, target_name


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols_ohe, categorical_cols_mte, numeric_cols, text_cols, passthrough_cols,
                 target_name, noise_k=0.006):
        self.categorical_cols_ohe = categorical_cols_ohe
        self.categorical_cols_mte = categorical_cols_mte
        self.numeric_cols = numeric_cols
        self.text_cols = text_cols  # Новая переменная для текстовых колонок
        self.passthrough_cols = passthrough_cols
        self.target_name = target_name
        self.noise_k = noise_k
        self.col_transform = None
        self.tfidf_vectorizers = {col: TfidfVectorizer() for col in
                                  self.text_cols}  # Инициализация TF-IDF векторизаторов

    def fit(self, X, y):
        cols_for_ohe_idx = [list(X.columns).index(col) for col in self.categorical_cols_ohe]
        cols_for_mte_idx = [list(X.columns).index(col) for col in self.categorical_cols_mte]
        numeric_cols_idx = [list(X.columns).index(col) for col in self.numeric_cols]
        passthrough_cols_idx = [list(X.columns).index(col) for col in self.passthrough_cols]

        """
        Параметры handle_unknown='ignore' и handle_unknown='impute'
        используются для обработки категорий в данных,
        которые не встречались в обучающем наборе (или в процессе fit).
        """

        """
        t = [
            ('OneHotEncoder', SklearnOneHotEncoder(handle_unknown='ignore', drop='first'), cols_for_ohe_idx),
            ('MeanTargetEncoder', TargetEncoder(handle_unknown='impute'), cols_for_mte_idx),
            ('StandardScaler', StandardScaler(), numeric_cols_idx),
            ('Passthrough', 'passthrough', passthrough_cols_idx)
        ]
        """

        # ('NumImputer', SimpleImputer(strategy='mean'), numeric_cols_idx),  # Импутация числовых данных
        # ('CatImputer', SimpleImputer(strategy='most_frequent'), cols_for_ohe_idx + cols_for_mte_idx),  # Импутация категориальных данных

        t = [
            ('OneHotEncoder', Pipeline(steps=[
                ('imputer_in', SimpleImputer(strategy='most_frequent')),  # заполняем пропуски до кодирования
                ('encoder', SklearnOneHotEncoder(handle_unknown='ignore', drop='first')),  # OHE
                ('imputer_out', SimpleImputer(strategy='most_frequent'))  # заполняем пропуски после кодирования
            ]), cols_for_ohe_idx),

            ('MeanTargetEncoder', Pipeline(steps=[
                ('imputer_in', SimpleImputer(strategy='most_frequent')),  # заполняем пропуски до кодирования
                ('encoder', TargetEncoder(handle_unknown='impute')),  # MTE
                ('imputer_out', SimpleImputer(strategy='most_frequent'))  # заполняем пропуски после кодирования
            ]), cols_for_mte_idx),

            ('StandardScaler', Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),  # заполняем пропуски
                ('scaler', StandardScaler())  # приводим значения к масштабу
            ]), numeric_cols_idx),

            ('Passthrough', 'passthrough', passthrough_cols_idx)  # Пропускаем без изменений
        ]

        self.col_transform = ColumnTransformer(transformers=t)
        self.col_transform.fit(X, y)

        # Фитинг TF-IDF векторизаторов на текстовых данных
        for col in self.text_cols:
            self.tfidf_vectorizers[col].fit(X[col])

        self.col_transform.fit(X, y)

        # Вызов функции для сохранения статистики по пользователям
        self._save_user_stats(X, y)

        return self

    def transform(self, X):
        X_transformed = self.col_transform.transform(X)

        X_transformed = pd.DataFrame(X_transformed)

        # Добавляем новые признаки, основанные на TF-IDF
        for col in self.text_cols:
            tfidf_data = self.tfidf_vectorizers[col].transform(X[col]).toarray()
            X_transformed[f'{col}_TotalTfIdf'] = tfidf_data.sum(axis=1)
            X_transformed[f'{col}_MaxTfIdf'] = tfidf_data.max(axis=1)
            X_transformed[f'{col}_MeanTfIdf'] = tfidf_data.mean(axis=1)
            X_transformed[f'{col}_MinTfIdf'] = tfidf_data.min(axis=1)

        # Вызов функции для создания дополнительных признаков
        X_transformed = self._create_additional_features(X, X_transformed)

        # Последняя сеть - назначаем имена столбцам --->>
        X_transformed.columns = self.get_feature_names_out()

        # Группировка по user_id и topic
        X_transformed = self._group_by_user_and_post(X, X_transformed)

        return X_transformed

    def _save_user_stats(self, X, y):
        X_with_target = pd.concat([X, y], axis=1)
        user_count_views = X_with_target.groupby('user_id').size()
        user_means = X_with_target.groupby('user_id')[self.target_name].mean()

        self.user_stats = {
            'views': user_count_views.to_dict(),
            'means': user_means.to_dict()
        }

    def _create_additional_features(self, X, X_transformed):
        user_count_views = pd.Series(self.user_stats['views'])
        user_means = pd.Series(self.user_stats['means'][self.target_name[0]])

        X_ = X.copy()
        X_.reset_index(inplace=True)

        ### Среднее кол-во просмотров
        X_transformed['userViews'] = X_['user_id'].map(user_count_views).fillna(user_count_views.mean())
        ### Средняя оценка
        X_transformed['userMeans'] = X_['user_id'].map(user_means).fillna(user_means.mean()) + np.random.normal(0,
                                                                                                                self.noise_k,
                                                                                                                X.shape[
                                                                                                                    0])

        X_transformed["user_id"] = X_["user_id"]

        X_transformed.set_index("user_id", inplace=True)

        return X_transformed

    def _group_by_user_and_post(self, X, X_transformed):
        # Определение OHE-столбцов
        ohe_columns = [col for col in X_transformed.columns if any(cat in col for cat in self.categorical_cols_ohe)]
        # Определение столбцов, над которыми не производились преобразования
        passthrough_columns = self.passthrough_cols

        # Столбцы для преобразования по методу аггрегации - мода
        mode_cols = ohe_columns + passthrough_columns

        # Создание словаря для агрегации: мода для OHE столбцов, среднее для остальных
        agg_dict = {}

        for col in X_transformed.columns:
            if col in mode_cols:
                agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan  # Мода для OHE столбцов
            elif col != 'user_id' and col != 'post_id':
                agg_dict[col] = 'mean'  # Среднее для остальных столбцов

        # Группировка по user_id и post_id с использованием словаря агрегации
        X_grouped = X_transformed.groupby(['user_id', 'post_id']).agg(agg_dict)

        X_grouped = X_grouped.drop(['post_id'], axis=1)

        # Сбрасываем индекс, сохраняя user_id и topic, но не добавляя их повторно в DataFrame
        X_grouped = X_grouped.reset_index()

        # Устанавливаем user_id в качестве индекса
        X_grouped.set_index("user_id", inplace=True)

        return X_grouped

    def get_feature_names_out(self):
        ohe_feature_names = self.col_transform.named_transformers_['OneHotEncoder'].get_feature_names_out(
            self.categorical_cols_ohe)
        mte_feature_names = self.categorical_cols_mte  # TargetEncoder не изменяет имена колонок
        numeric_feature_names = self.numeric_cols
        passthrough_feature_names = self.passthrough_cols
        additional_feature_names = ['userViews', 'userMeans']
        tfidf_feature_names = [f'{col}_TotalTfIdf' for col in self.text_cols] + \
                              [f'{col}_MaxTfIdf' for col in self.text_cols] + \
                              [f'{col}_MeanTfIdf' for col in self.text_cols] + \
                              [f'{col}_MinTfIdf' for col in self.text_cols]

        return np.concatenate([ohe_feature_names, mte_feature_names, numeric_feature_names, passthrough_feature_names,
                               additional_feature_names, tfidf_feature_names])


def group_target(df, target_name):
    # Группировка по user_id и post_id с агрегацией по моде для target_name
    df_grouped = (
        df.groupby(['user_id', 'post_id'])[target_name]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .drop('post_id', axis=1)
        .set_index('user_id')
    )
    return df_grouped

from tqdm import tqdm
from sklearn.pipeline import Pipeline


def preprocess_data(engine, chunksize=50000, feed_data_limit=5000, ohe_threshold=5):
    # 1. Вытягиваем данные из БД и сохраняем в локальном репозитории
    print("Загрузка начальных данных...")
    initial_table = load_initial_tables(engine=engine, chunksize=chunksize, feed_data_limit=feed_data_limit)
    print("1. Начальные данные загружены!")

    # 2. Подготовка данных для трейн и тест
    print("Подготовка данных для обучения...")
    data, X_data, y_data, categorical_cols, numeric_cols, text_cols, passthrough_cols, target_name = prepare_data(
        initial_table)

    # 3. Преобразование признаков
    cols_for_ohe = [x for x in categorical_cols if X_data[x].nunique() < ohe_threshold]
    cols_for_mte = [x for x in categorical_cols if X_data[x].nunique() >= ohe_threshold]

    print("Настройка пайплайна...")
    pipeline = Pipeline([
        ('custom_transformer', CustomTransformer(
            categorical_cols_ohe=cols_for_ohe,
            categorical_cols_mte=cols_for_mte,
            numeric_cols=numeric_cols,
            text_cols=text_cols,
            passthrough_cols=passthrough_cols,
            target_name=target_name
        ))
    ])

    pipeline.fit(X_data, y_data)

    X_transformed = pipeline.transform(X_data)

    # y_transformed = group_target(pd.concat([X_data, y_data], axis=1), target_name)

    # df_transformed = pd.concat([X_transformed, y_transformed], axis=1)

    # Возвращаем обработанные данные
    return X_transformed


def load_to_sql(table_name, data):
    ### 6. Записываем DataFrame в базу данных -->>

    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")
    data.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=10000)


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


if __name__ == "__main__":
    # Препроцессинг данных:
    engine = get_engine()
    data = preprocess_data(
        engine,
        chunksize=50000,
        feed_data_limit=100000,
        ohe_threshold=5
    )
    # Загружаем в БД таблицу с созданными признаками
    table_name = 'danil_temnkhudov_features_lesson_22_all'
    load_to_sql(table_name, data)