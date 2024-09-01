import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from tqdm import tqdm
import gc


def get_engine():
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    return create_engine(SQLALCHEMY_DATABASE_URL)


def execute_query(engine, query, chunksize=50000):
    with engine.connect() as connection:
        for chunk in pd.read_sql(query, connection, chunksize=chunksize):
            yield chunk


def load_and_save_table(engine, table_name, save_file_path, chunksize):
    data_chunks = []
    query = f"SELECT * FROM public.{table_name}"
    for chunk in tqdm(execute_query(engine, query, chunksize=chunksize), desc=f"Loading {table_name}..."):
        data_chunks.append(chunk)
        del chunk
        gc.collect()
    data = pd.concat(data_chunks)
    save_to_csv(data, save_file_path, chunksize)
    return data


def save_to_csv(data, filepath, chunksize=50000):
    for i in tqdm(range(0, data.shape[0], chunksize), desc=f"Saving {filepath} to CSV..."):
        chunk = data.iloc[i:i + chunksize]
        if i == 0:
            chunk.to_csv(filepath, index=False, mode='w', sep=',')
        else:
            chunk.to_csv(filepath, index=False, mode='a', header=False, sep=',')
        del chunk
        gc.collect()


def load_initial_tables(engine, save_file_path, chunksize=50000):
    user_data = load_and_save_table(engine, "user_data", save_file_path + "user_data.csv", chunksize)
    post_text_df = load_and_save_table(engine, "post_text_df", save_file_path + "post_text_df.csv", chunksize)
    feed_data = load_and_save_table(engine, "feed_data", save_file_path + "feed_data.csv", chunksize)

    # Объединение таблиц
    data = feed_data.merge(user_data, on='user_id', how='inner')
    data = data.merge(post_text_df, on='post_id', how='inner')

    ### Отсортируем данные по дате
    print("Сортировка данных по дате: ")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data.reset_index(drop=True, inplace=True)

    data.drop(["text"], axis=1, inplace=True)

    save_to_csv(data, save_file_path + "all_data.csv", chunksize)
    return data


def train_test_split_sorted(data, train_size=0.8):
    ### Делим выборку 80 на 20
    split_index = int(len(data) * train_size)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    print("Предварительная выборка на трейн: ")
    print(train)
    print("Предварительная выборка на тест: ")
    print(test)

    return train, test


def create_features(
        data,  # Сырые данные выгруженные из БД
        ohe_threshold=5,  # Отсечка по уникальным значениям для кодирования фичей по ohe или счетчикам
        tf_idf_features=0,  # Задаем желаемое кол-во фичей для кодирования текста
        like_target_threshold=0.5  # Отсечка для бинаризации таргетной переменной
):
    print("Сырые данные выгруженные из БД: ")
    print(data)

    ### 1. Кодирование имеющихся признаков / OneHotEncoding // Counters+Noise -->>

    # Список категориальных признаков
    categorical_features = ["gender", "country", "city", "exp_group", "os", "source", "topic"]

    # Создаем таргет like_target - признак того, что юзер лайкнул пост или нет
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
                data[col] = data[col].map(mean_encoding)  # + noise
                encoded_cols.append(col)
        return data, encoded_cols

    # Применяем OHE или счетчики к категориальным признакам
    data, encoded_cols = encode_features(data, categorical_features, ohe_threshold)

    ### 2. TF-IDF (кодирование текста) -->>

    # TF-IDF кодирование текста
    if tf_idf_features > 0:
        tfidf_vectorizer = TfidfVectorizer(max_features=tf_idf_features)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        data = pd.concat([data.drop(columns=['text']), tfidf_df], axis=1)
    else:
        if 'text' in data.columns:
            data.drop(columns=['text'], inplace=True)

    ### 3. Создание дополнительных признаков -->>

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
        'userViews': 'mean',  # среднее количество просмотров
        'userMeans': 'mean',  # средняя оценка
    }

    # Добавляем дополнительные признаки, если есть
    for col in data.columns:
        if col not in aggregation_dict and col != 'user_id' and col != 'topic' and col != "timestamp":
            aggregation_dict[col] = 'mean'

    data = data.groupby(['user_id', 'topic']).agg(aggregation_dict).reset_index()

    data["like_target"] = data["like_target"].apply(lambda x: 0 if x < like_target_threshold else 1)

    ### 5. Финальная доработка -->>

    # Дропаем ненужные колонки
    ### data.drop(["like_target"], axis=1, inplace=True)

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


def load_manager(
        train_data,
        test_data,
        file_path
):
    for data in ["train", "test"]:
        if data == "train":
            initial_table = train_data
            table_name = "train"
        else:
            initial_table = test_data
            table_name = "test"

        print(f"Создание выборки для {table_name}: ")

        # 2. Создаем признаки
        features = create_features(
            data=initial_table,  # сырые данные выгруженные из БД
            ohe_threshold=5,  # отсечка по уникальным значениям для кодирования фичей по ohe или счетчикам
            tf_idf_features=0,  # Задаем желаемое кол-во фичей для кодирования текста
            like_target_threshold=0.5  # Отсечка для бинаризации таргетной переменной
        )
        print("2. Признаки созданы!")

        save_to_csv(
            data=features,
            filepath=file_path + f"features_{table_name}",
            chunksize=50000
        )

        # 3. Загружаем в БД таблицу с созданными признаками
        table_name = f'danil_temnkhudov_features_lesson_22_for_model_{table_name}'
        load_to_sql(table_name, features)
        print("3. Таблица с созданными признаками загружена в БД!")

        # 4. Выгружаем из БД сформированные признаки
        features = load_features(table_name=table_name)
        print("4. Таблица с созданными признаками выгружена из БД: ")
        print(features)


if __name__ == "__main__":

    # 1. Вытягиваем данные из БД и сохранение в локальном репозитории
    engine = get_engine()
    chunksize = 50000
    path_data = "C:/python_projects/FinalProject_2/data/"
    # initial_table = load_initial_tables(engine=engine, save_file_path=path_data, chunksize=50000)
    initial_table = pd.read_csv("C:/python_projects/FinalProject_2/data/all_data.csv", sep=",")
    print("1. Начальные данные загружены!")

    # 1.1 Разделим данные на трейн и тест в соотношении 80 на 20
    train_data, test_data = train_test_split_sorted(initial_table, train_size=0.8)

    load_manager(train_data, test_data, file_path=path_data)