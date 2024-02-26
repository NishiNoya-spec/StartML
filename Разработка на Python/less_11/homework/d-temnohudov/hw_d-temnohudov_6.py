from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from textwrap import dedent

# Определение аргументов по умолчанию для DAG
default_args = {
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Создание DAG с указанием его настроек
with DAG(
    'hw_d-temnohudov_6',
    default_args=default_args,
    description='hw_d-temnohudov_6',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['hw_d-temnohudov_6'],
) as dag:

    for i in range(10):
        t1 = BashOperator(
            task_id=f'bash_task_{i}',
            bash_command='echo $NUMBER',
            env={'NUMBER': str(i)}, 
            dag=dag,
        )
        
    t1