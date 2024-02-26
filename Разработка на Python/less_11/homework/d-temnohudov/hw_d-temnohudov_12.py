from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable
import psycopg2

def print_variable_value():
    is_startml = Variable.get("is_startml")
    print(f"Value of 'is_startml' variable: {is_startml}")

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
    'hw_d-temnohudov_12',
    default_args=default_args,
    description='hw_d-temnohudov_12',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['hw_d-temnohudov_12'],
) as dag:

    # Создание оператора PythonOperator
    t1 = PythonOperator(
        task_id='print_variable_value',
        python_callable=print_variable_value,
        provide_context=True, 
    )

    t1

