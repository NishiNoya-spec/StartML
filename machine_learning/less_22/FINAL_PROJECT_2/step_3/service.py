import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
import pickle


app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("CatBoost.cbm")
    file_model = CatBoostClassifier()
    file_model.load_model(model_path, format="cbm")
    return file_model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():
    df_load_features = batch_load_sql('''SELECT * FROM danil_temnkhudov_features_lesson_22 ''')
    return(df_load_features)


def load_posts():
    df_load_posts = batch_load_sql('''SELECT * FROM "post_text_df" ''')
    return(df_load_posts)


load_model = CatBoostClassifier()
load_model = load_models()
load_features_df = load_features()
load_posts_df = load_posts()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:

    df_feat_user = load_features_df[load_features_df['user_id'] == id]

    df_feat_user_pred = df_feat_user.drop(['user_id', 'post_id'], axis=1)
    df_feat_user['predict'] = load_model.predict_proba(df_feat_user_pred)[:, 1]

    df_feat_user = df_feat_user.sort_values('predict', ascending=False)[:5]
    rec_posts = df_feat_user['post_id'].to_list()

    result_rec_df = load_posts_df[load_posts_df['post_id'].isin(rec_posts)].rename(columns={'post_id': 'id'})
    result = result_rec_df.to_dict(orient='records')
    return result
