from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

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
    'hw_d-temnohudov_2',
    default_args=default_args,
    description='airflow - hw_2',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['airflow_hw_2'],
) as dag:
    for i in range(30):
        # Если номер задачи меньше 10, создаем BashOperator
        if i < 10:
            t1 = BashOperator(
                task_id=f'bash_task_{i}',
                bash_command=f'echo "Bash Task {i}"',
                dag=dag,
            )
        # Если номер задачи больше или равен 10, создаем PythonOperator
        else:
            def print_task_number(task_number):
                print(f"Task number is: {task_number}")
            
            t2 = PythonOperator(
                task_id=f'python_task_{i}',
                python_callable=print_task_number,
                op_kwargs={'task_number': i},
                dag=dag,
            )
        
        if i >= 10:
            t1 >> t2
