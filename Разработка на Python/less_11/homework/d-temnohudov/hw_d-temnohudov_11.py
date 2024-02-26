from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
import psycopg2

def find_top_liked_user():
    # Получение подключения к базе данных
    conn_id = "startml_feed"
    creds = BaseHook.get_connection(conn_id)
    conn_string = f"postgresql://{creds.login}:{creds.password}@{creds.host}:{creds.port}/{creds.schema}"

    # Подключение к базе данных и выполнение запроса
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cursor:
            # Выполняем запрос для нахождения пользователя с наибольшим количеством лайков
            query = """
                SELECT user_id, COUNT(action) AS like_count
                FROM feed_action
                WHERE action = 'like'
                GROUP BY user_id
                ORDER BY like_count DESC
                LIMIT 1;
            """
            cursor.execute(query)
            result = cursor.fetchone()

            # Возвращаем результат в виде словаря
            if result:
                user_id, like_count = result
                return {'user_id': user_id, 'count': like_count}
            else:
                return None

# Определение аргументов по умолчанию для DAG
default_args = {
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Создание DAG с указанием его настроек
with DAG(
    'hw_d-temnohudov_10',
    default_args=default_args,
    description='hw_d-temnohudov_10',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['hw_d-temnohudov_10'],
) as dag:

    # Создание оператора PythonOperator
    t1 = PythonOperator(
        task_id='find_top_liked_user_task',
        python_callable=find_top_liked_user,
        provide_context=True, 
    )

    t1

