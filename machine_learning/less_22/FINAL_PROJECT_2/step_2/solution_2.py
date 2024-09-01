import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine

def load_initial_tables():

    ### 1. Вытягиваем данные из БД
    ### и создаем начальную таблицу для последующего кодирования признаков -->>

    # Создаем объект engine
    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")

    # Функция для загрузки данных из базы
    def load_data(table_name, limit=None):
        query = f"SELECT * FROM public.{table_name}"
        if limit:
            query += f" LIMIT {limit}"
        with engine.connect() as connection:
            data = pd.read_sql(query, con=connection)
        return data

    # Загрузка данных из таблиц
    user_data = load_data("user_data")
    post_text_df = load_data("post_text_df")
    feed_data = load_data("feed_data", limit=1000000)  # используем лимит для уменьшения объема данных

    # Объединение таблиц
    data = feed_data.merge(user_data, on='user_id', how='inner')
    data = data.merge(post_text_df, on='post_id', how='inner')

    return data


def create_features(data):

    ### 2. OneHotEncoding // Counters+noise -->>

    ohe_threshold = 2  # отсечка по уникальным значениям для кодирования фичей по ohe или счетчикам

    # Список категориальных признаков
    categorical_features = ["gender", "country", "city", "exp_group", "os", "source", "topic"]

    # Создаем признак like_target
    data["like_target"] = data.apply(lambda row: 1 if row['action'] == 'like' or row['target'] == 1 else 0, axis=1)
    data.drop(["action", "target"], axis=1, inplace=True)

    # Функция для кодирования признаков
    def encode_features(data, categorical_features, ohe_threshold):
        encoded_cols = []
        for col in categorical_features:
            if data[col].nunique() <= ohe_threshold:
                one_hot = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat((data.drop(col, axis=1), one_hot), axis=1)
                encoded_cols.extend(one_hot.columns)
            else:
                mean_encoding = data.groupby(col)['like_target'].mean()
                noise = np.random.normal(0, 0.1, data.shape[0])
                data[col] = data[col].map(mean_encoding) + noise
                encoded_cols.append(col)
        return data, encoded_cols

    # Применяем OHE или счетчики к категориальным признакам
    data, encoded_cols = encode_features(data, categorical_features, ohe_threshold)

    ### 3. TF-IDF (кодирование текста) -->>

    # TF-IDF кодирование текста
    tf_idf_features = 0  # Задаем желаемое кол-во фичей для кодирования текста

    if tf_idf_features > 0:
        tfidf_vectorizer = TfidfVectorizer(max_features=tf_idf_features)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        data = pd.concat([data.drop(columns=['text']), tfidf_df], axis=1)
    else:
        if 'text' in data.columns:
            data.drop(columns=['text'], inplace=True)

    ### 4. Расчет дополнительных признаков -->>

    # Посчитаем среднее кол-во просмотров всех юзеров
    user_count_views = data.groupby('user_id').size()
    data['userViews'] = data['user_id'].map(user_count_views)

    # И среднюю оценку по средним оценкам всех юзеров
    user_means = data.groupby('user_id')['like_target'].sum()
    noise = np.random.normal(0, 0.1, data.shape[0])
    data['userMeans'] = data['user_id'].map(user_means) + noise

    overall_views_mean = user_count_views.mean()
    overall_meanrating_mean = user_means.mean()

    data['userViews'] = data['user_id'].map(user_count_views).fillna(overall_views_mean)
    data['userMeans'] = data['user_id'].map(user_means).fillna(overall_meanrating_mean)

    ### --++

    # Агрегируем данные по пользователю
    aggregation_dict = {
        'post_id': 'count',  # количество постов
        'userViews': 'mean',  # среднее количество просмотров
        'userMeans': 'mean',  # средняя оценка
    }

    # Добавляем агрегации для закодированных признаков
    for col in encoded_cols:
        aggregation_dict[col] = 'mean'

    # Добавляем дополнительные признаки, если есть
    for col in data.columns:
        if col not in aggregation_dict and col != 'user_id':
            aggregation_dict[col] = 'mean'

    data = data.groupby('user_id').agg(aggregation_dict).reset_index()

    ### 5. Финальная доработка -->>

    # Дропаем ненужные колонки
    data.drop(["like_target", "timestamp"], axis=1, inplace=True)

    ### data.set_index("user_id", inplace=True)

    return data


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



# 1. Вытягиваем данные из БД
initial_table = load_initial_tables()
# 2. Создаем признаки
features = create_features(initial_table)
# 3. Загружаем в БД таблицу с созданными признаками
table_name = 'danil_temnkhudov_features_lesson_22'
load_to_sql(table_name, features)
# 4. Выгружаем из БД сформированные признаки
features = load_features(table_name=table_name) 
