from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

# Add your project path so Airflow can find 'train.py'
sys.path.append('/Users/zacharymyrick/.jenkins/workspace/docker-hello-world')
from train import train_model 

default_args = {
    'owner': 'zachary',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'collective_memory_pipeline',
    default_args=default_args,
    description='Automated Training Pipeline for Housing Prices',
    schedule_interval='@daily', 
    start_date=datetime(2026, 2, 1),
    catchup=False,
) as dag:

    # Task 1: The Core Training Step
    # This replaces the manual "Retrain" button on your dashboard
    train_task = PythonOperator(
        task_id='train_and_register_mlflow',
        python_callable=train_model,
    )

    # Future Task placeholders for our "Longview" plan:
    # dbt_transform >> train_task >> slack_notification