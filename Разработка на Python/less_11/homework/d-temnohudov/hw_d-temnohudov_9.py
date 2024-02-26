from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from textwrap import dedent

import requests
import json


def xcom_put_data(ti):
    ti.xcom_push(
        key='sample_xcom_key',
        value='xcom test',
    )

def xcom_fetch_data(ti):
    result = ti.xcom_pull(
        key='sample_xcom_key',
        task_ids = 'xcom_put_data',
    )
    print(f"result: {result}")


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
    "hw_d-temnohudov_9",
    default_args=default_args,
    description="hw_d-temnohudov_9",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=["hw_d-temnohudov_9"],
) as dag:

    t1 = PythonOperator(
        task_id = 'xcom_put_data',
        python_callable=xcom_put_data,
    )
    t2 = PythonOperator(
        task_id = 'xcom_fetch_data',
        python_callable=xcom_fetch_data,
    )

    t1 >> t2
