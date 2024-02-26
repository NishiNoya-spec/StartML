from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from textwrap import dedent

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
    "hw_d-temnohudov_7",
    default_args=default_args,
    description="hw_d-temnohudov_7",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=["hw_d-temnohudov_7"],
) as dag:

    ts = "{{ ts }}"
    run_id = "{{ run_id }}"

    for i in range(30):
        # Если номер задачи меньше 10, создаем BashOperator
        if i < 10:
            t1 = BashOperator(
                task_id=f"bash_task_{i}",
                bash_command=f'echo "Bash Task {i}"',
                dag=dag,
            )
        # Если номер задачи больше или равен 10, создаем PythonOperator
        else:

            def print_task_number(ts, run_id, **kwargs):
                print(f"Task number is: {kwargs['task_number']}")
                print(f"Timestamp is: {ts}")
                print(f"Run id is: {run_id}")

            t2 = PythonOperator(
                task_id=f'python_task_{i}',
                python_callable=print_task_number,
                op_kwargs={"task_number": i},
                provide_context=True,  # Передача контекста
                dag=dag,
            )

            if i >= 10:
                t1 >> t2
