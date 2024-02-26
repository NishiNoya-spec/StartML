from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
import psycopg2


def decide_path(**kwargs):
    is_startml = Variable.get("is_startml")
    if is_startml == "True":
        return "startml_desc"
    else:
        return "not_startml_desc"


def startml_desc(**kwargs):
    print("StartML is a starter course for ambitious people")


def not_startml_desc(**kwargs):
    print("Not a startML course, sorry")


# Определение аргументов по умолчанию для DAG
default_args = {
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Создание DAG с указанием его настроек
with DAG(
    "hw_d-temnohudov_13",
    default_args=default_args,
    description="hw_d-temnohudov_13",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 2, 24),
    catchup=False,
    tags=["hw_d-temnohudov_13"],
) as dag:

    before_branching = DummyOperator(
        task_id="before_branching"
    )

    determine_course = BranchPythonOperator(
        task_id="determine_course",
        python_callable=decide_path,
        provide_context=True,
    )

    startml_desc = PythonOperator(
        task_id="startml_desc",
        python_callable=startml_desc,
        provide_context=True,
    )

    not_startml_desc = PythonOperator(
        task_id="not_startml_desc",
        python_callable=not_startml_desc,
        provide_context=True,
    )

    after_branching = DummyOperator(
        task_id="after_branching"
    )


    before_branching >> determine_course >> [startml_desc, not_startml_desc] >> after_branching
