from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
training_pipeline=None
# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

# [END imporETL DAG tutorial_prediction',
    # [START default_args]
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
with DAG(
    'recommender-training',
    default_args={'retries': 2}, #pipeline if fails once will retty once again
    # [END default_args]
    description='Course Recommendation Pipeline Project',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 2, 23, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # [START extract_function]

    from src.configurations.training_config import RecommenderConfig
    from src.pipeline.train_pipeline import TrainingPipeline
    from src.entity.config_entity import *
    training_pipeline = TrainingPipeline(RecommenderConfig())

    def data_ingestion(**kwargs):
        from src.entity.artifact_entity import DataIngestionArtifact,\
        ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
        ti = kwargs['ti']
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        print(data_ingestion_artifact)
        ti.xcom_push('data_ingestion_artifact', data_ingestion_artifact)

    def model_trainer(**kwargs):
        from src.entity.artifact_entity import DataIngestionArtifact,\
        ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
        ti  = kwargs['ti']

        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
        data_ingestion_artifact=DataIngestionArtifact(*(data_ingestion_artifact))

        model_trainer_artifact=training_pipeline.start_model_trainer(data_ingestion_artifact=data_ingestion_artifact)

        ti.xcom_push('model_trainer_artifact', model_trainer_artifact._asdict())

    def model_evaluation(**kwargs):
        from src.entity.artifact_entity import DataIngestionArtifact,\
        ModelTrainerArtifact,ModelEvaluationArtifact
        ti  = kwargs['ti']
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
        data_ingestion_artifact=DataIngestionArtifact(*(data_ingestion_artifact))

        model_trainer_artifact = ti.xcom_pull(task_ids="model_trainer",key="model_trainer_artifact")
        print(model_trainer_artifact)
        model_trainer_artifact=ModelTrainerArtifact.construct_object(**model_trainer_artifact)

        model_evaluation_artifact = training_pipeline.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
    
        ti.xcom_push('model_evaluation_artifact', model_evaluation_artifact.to_dict())

    def push_model(**kwargs):
        from src.entity.artifact_entity import DataIngestionArtifact,\
        ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
        ti  = kwargs['ti']
        model_evaluation_artifact = ti.xcom_pull(task_ids="model_evaluation",key="model_evaluation_artifact")
        model_evaluation_artifact=ModelEvaluationArtifact(*(model_evaluation_artifact))
        
        if model_evaluation_artifact.is_model_accepted:
            model_pusher_artifact = training_pipeline.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
            print(f'Model pusher artifact: {model_pusher_artifact}')
        else:
            print("Trained model rejected.")
            print("Trained model rejected.")
        print("Training pipeline completed")



    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
    )

    data_ingestion.doc_md = dedent(
        """\
    #### Extract task
    A simple Extract task to get data ready for the rest of the data pipeline.
    In this case, getting data is simulated by reading from a hardcoded JSON string.
    This data is then put into xcom, so that it can be processed by the next task.
    """
    )

    model_trainer = PythonOperator(
        task_id="model_trainer", 
        python_callable=model_trainer

    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation", python_callable=model_evaluation
    )   

    push_model = PythonOperator(
            task_id="push_model",
            python_callable=push_model

    )

    data_ingestion >> model_trainer >> model_evaluation >> push_model