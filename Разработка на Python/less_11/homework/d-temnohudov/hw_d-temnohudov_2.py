from datetime import datetime, timedelta
from textwrap import dedent

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow .operators.python import PythonOperator


with DAG(
    'hw_d-temnohudov_2',

    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),  # timedelta из пакета datetime
    },

    description='airflow - hw_2',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['airflow_hw_2'],

) as dag:

    t1 = BashOperator(
        task_id='print_pwd',
        bash_command='pwd',
    )

    def print_logical_date(ds):
        print(ds)

    t2 = PythonOperator(
        task_id='print_logical_date',
        python_callable=print_logical_date,
    )

    t1 >> t2




