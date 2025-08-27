from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data.load_data import load_imdb_data
from src.features.text_cleaning import DataTransformation
from src.pipelines.train import ModelTrainer
from src.evaluation.evaluate_model import Evaluater
from src.features.build_feature import FeatureBuilder
import yaml


def ingest_data():
    """Ingest raw data"""
    print("Data Ingestion Pipeline Started Successfully!")
    df = load_imdb_data("raw")
    print("Data Ingestion Pipeline Started Successfully!")
    print(f"Data shape: {df.shape}")



def preprocess_data():
    """Preprocess text data"""
    print("Data Pre-processing Pipeline Started Successfully!")

    transformer = DataTransformation()
    transformer.preprocess_data()

    builder = FeatureBuilder()
    builder.build_tfidf_features()


def train_model():
    """Train sentiment model"""
    print("Model Training Pipeline Started Successfully!")
    trainer = ModelTrainer()
    trainer.train_model()


def evaluate_model():
    """Evaluate multiple models + log MLflow"""
    print("Model Evaluation Pipeline Started Successfully!")

    evaluater = Evaluater()
    evaluater.evaluate_and_experiment()


# Define DAG
with DAG(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 8, 18),
    schedule_interval=None,  # Run manually for now
    catchup=False,
    tags=["mlops", "imdb", "nlp"],
) as dag:

    ingest_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
    )

    preprocess_task = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_model,
    )

    # Define pipeline order
    ingest_task >> preprocess_task >> train_task >> evaluate_task
