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
    'hw_d-temnohudov_5',
    default_args=default_args,
    description='hw_d-temnohudov_5',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=['hw_d-temnohudov_5'],
) as dag:
    
    templated_command = dedent(
    """
    {% for i in range(5) %}
        echo "{{ ts }}"
        echo "{{ run_id }}"
    {% endfor %}
    """
    )  # поддерживается шаблонизация через Jinja

    t1 = BashOperator(
        task_id='templated',
        depends_on_past=False,
        bash_command=templated_command,
        dag=dag,
        env={'ts': '{{ ts }}', 'run_id': '{{ run_id }}'}  # Передача значений ts и run_id в команду
    )