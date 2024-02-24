from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

app = FastAPI()


@app.get("/")
def say_hello():
    return "hello"


@app.get("/sub")
def subtruct(a: int, b: int) -> int:
    return a - b


@app.get("/add/{num1}/{num2}")
def addanother(num1: int, num2: int):
    return num1 + num2


@app.post("/user")
def print(name: str):
    return {"message": f"hello, {name}"}


@app.get("/booking/all")
def all_bookings():
    conn = psycopg2.connect(
        "postgresql://postgres:password@localhost:5432/exercises",
        cursor_factory=RealDictCursor,
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM cd.bookings
        """
    )
    return cursor.fetchall()


class UserGet(BaseModel):
    """
    {
    "id": 200,
    "gender": 1,
    "age": 34,
    "country": "Russia",
    "city": "Degtyarsk",
    "exp_group": 3,
    "os": "Android",
    "source": "ads"
    }
    """

    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    # class Config нужен для того чтобы обращаться к свойствам объекта
    # таким образом UserGet.id
    # и таким образом UserGet['id']

    class Config:
        orm_mode = True


@app.get("/users/all", response_model=List[UserGet])
def all_users():

    def fetch_query(query):

        connection = psycopg2.connect(
            database="startml",
            user="robot-startml-ro",
            password="pheiph0hahj1Vaif",
            host="postgres.lab.karpov.courses",
            port=6432,
            cursor_factory=RealDictCursor,
        )

        cursor = connection.cursor()

        cursor.execute(f"""{query}""")

        results = cursor.fetchall()

        cursor.close()

        connection.close()

        return results

    query = """
            SELECT *
            FROM "user" 
        """
    df_user = fetch_query(query)

    result = df_user

    logger.info(result)

    return result


class SimpleUser(BaseModel):
    name: str
    surname: str


@app.post("/user/validate")
def validate_user(user: SimpleUser):
    return "ok"
