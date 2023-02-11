import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from src.exception import TrainException
from src.logger import logging
from src.constants.cloud_constants import S3_TRAINING_BUCKET_NAME
from src.constants.training_pipeline import SAVED_MODEL_DIR
from src.utils.s3_syncer import S3Sync
from src.configurations.training_config import RecommenderConfig
class TrainingPipeline:
    def __init__(self, recommender_config = RecommenderConfig()):
        self.training_pipeline_config = recommender_config
        self.artifact_dir = self.training_pipeline_config.pipeline_config.artifact_dir
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
              "Entered the start_data_ingestion method of TrainPipeline class"
            )
            self.data_ingestion_config = self.training_pipeline_config.get_data_ingestion_config()
            
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
    
        except Exception as e:
            raise TrainException(e, sys)

    def start_model_trainer(self,data_ingestion_artifact:DataIngestionArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            model_trainer_config = self.training_pipeline_config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config, data_ingestion_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()


            logging.info("Performed the Model Training operation")
            logging.info(
                "Exited the start_model_trainer method of TrainPipeline class"
            )

            return model_trainer_artifact

        except  Exception as e:
            raise  TrainException(e,sys)

    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact
                                ) -> ModelEvaluationArtifact:
        try:
            logging.info("Entered the start_model_evaluation method of TrainPipeline class")
            model_eval_config = self.training_pipeline_config.get_model_evaluation_config()
            model_eval = ModelEvaluation(model_eval_config, data_ingestion_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()

            logging.info("Performed the Model Evaluation operation")
            logging.info(
                "Exited the start_model_evaluation method of TrainPipeline class"
            )
            return model_eval_artifact
        except  Exception as e:
            raise  TrainException(e,sys)

    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logging.info("Entered the start_model_pusher method of TrainPipeline class")
            model_pusher_config = self.training_pipeline_config.get_model_pusher_config()
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Performed the Model Pusher operation")
            logging.info(
                "Exited the start_model_pusher method of TrainPipeline class"
            )

            return model_pusher_artifact
        except  Exception as e:
            raise TrainException(e,sys)
    
    def sync_artifact_dir_to_s3(self):
        try:
            logging.info("Entered the sync_artifact_dir_to_s3 method of TrainPipeline class")
            aws_bucket_url = f"s3://{S3_TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.artifact_dir,aws_buket_url=aws_bucket_url)
            logging.info("Performed Syncing of artifact to S3 bucket")

        except Exception as e:
            raise TrainException(e,sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            logging.info("Entered the sync_saved_model_dir_to_s3 method of TrainPipeline class")
            aws_bucket_url = f"s3://{S3_TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR,aws_buket_url=aws_bucket_url)
            logging.info("Performed Syncing of saved models to S3 bucket")
        except Exception as e:
            raise TrainException(e,sys)

    def run_pipeline(self,) -> None:
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_ingestion_artifact = self.start_data_ingestion()
            model_trainer_artifact = self.start_model_trainer(data_ingestion_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact, model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                print("Process Completed Succesfully. Model Trained and Evaluated but the Trained model is not better than the best model. So, we do not push this model to Production. Exiting.")
                self.sync_artifact_dir_to_s3()
                raise Exception("Model Not Pushed to Production")
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            logging.info("Training Pipeline Running Operation Complete")
            logging.info(
                "Exited the run_pipeline method of TrainPipeline class"
            )
        except Exception as e:
            #self.sync_artifact_dir_to_s3()
            raise TrainException(e, sys)
