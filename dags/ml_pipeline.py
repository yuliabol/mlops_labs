import os
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.python import PythonSensor
import mlflow

# Configuration
ML_PROJECT_PATH = "/opt/airflow/ml_project"
METRICS_FILE = os.path.join(ML_PROJECT_PATH, "metrics.json")
F1_THRESHOLD = 0.3

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_data_ready():
    train_path = os.path.join(ML_PROJECT_PATH, "data/prepared/train.csv")
    return os.path.exists(train_path)

def evaluate_model_performance(**kwargs):
    if not os.path.exists(METRICS_FILE):
        return "stop_pipeline"
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    
    f1_score = metrics.get("test_f1", 0)
    print(f"Model F1 Score: {f1_score} (Threshold: {F1_THRESHOLD})")
    
    if f1_score >= F1_THRESHOLD:
        return "register_model"
    else:
        return "stop_pipeline"

def register_model_in_mlflow(**kwargs):
    print("Registering model in MLflow Model Registry...")
    # mlflow.register_model(model_uri="runs:/...", name="StrokePredictionModel")
    pass

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline with DVC and MLflow',
    schedule=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'lab5'],
) as dag:

    wait_for_data = PythonSensor(
        task_id='wait_for_data',
        python_callable=check_data_ready,
        timeout=600,
        poke_interval=30,
        mode='poke',
    )

    data_prep = BashOperator(
        task_id='data_preparation',
        bash_command=f'cd {ML_PROJECT_PATH} && dvc repro prepare',
    )

    train_model = BashOperator(
        task_id='model_training',
        bash_command=f'cd {ML_PROJECT_PATH} && python src/train.py data/prepared/train.csv data/prepared/test.csv data/models --ci',
    )

    evaluate_branch = BranchPythonOperator(
        task_id='evaluate_and_branch',
        python_callable=evaluate_model_performance,
    )

    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model_in_mlflow,
    )

    stop_pipeline = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Model quality insufficient. Pipeline stopped."',
    )

    wait_for_data >> data_prep >> train_model >> evaluate_branch
    evaluate_branch >> register_model
    evaluate_branch >> stop_pipeline
