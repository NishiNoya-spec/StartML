import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine

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


def load_features() -> pd.DataFrame:

    ### 8. Читаем DataFrame из базы данных -->>
    
    table_name="danil_temnkhudov_features_lesson_22"
    query = f"SELECT * FROM {table_name}"
    return batch_load_sql(query)

