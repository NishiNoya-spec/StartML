from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from textwrap import dedent

import requests
import json


def return_str():
    return('Airflow tracks everything')

def get_str(ti):
    airflow_string = ti.xcom_pull(
        key='return_value',
        task_ids='return_str',
    )
    print(f"Recieved string from XCom: {airflow_string}")


# Определение аргументов по умолчанию для DAG
default_args = {
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Создание DAG с указанием его настроек
with DAG(
    "hw_d-temnohudov_10",
    default_args=default_args,
    description="hw_d-temnohudov_10",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=["hw_d-temnohudov_10"],
) as dag:

    t1 = PythonOperator(
        task_id = 'return_str',
        python_callable=return_str,
    )
    t2 = PythonOperator(
        task_id = 'get_str',
        python_callable=get_str,
    )

    t1 >> t2
