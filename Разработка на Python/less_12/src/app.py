from fastapi import FastAPI, Depends
import psycopg2
import yaml
import uvicorn
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from src.crud import get_likes, get_feed
from pathlib import Path

app = FastAPI()



def get_db():
    try:
        conn = psycopg2.connect(
            database=os.environ["POSTGRES_DATABASE"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=os.environ["POSTGRES_PORT"]
        )
        yield conn
    finally:
        conn.close()


def config():
    print(__file__)
    with open(Path(__file__).parent.parent / "params.yaml", "r") as f:
        return yaml.safe_load(f)


@app.get("/user/feed")
def get_all_users(user_id: int, limit: int = 10, db = Depends(get_db), config: dict = Depends(config)):
    with db.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT *
            FROM feed_action
            WHERE user_id = %(user_id)s
                AND time >= %(start_date)s
            ORDER BY time
            LIMIT %(limit_user)s
            """,
            {"user_id": user_id, "limit_user": limit, "start_date": config["feed_start_date"]},
        )
        return cursor.fetchall()


if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app)

