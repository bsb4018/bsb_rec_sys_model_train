from datetime import datetime
import os
from src.constants import training_pipeline


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.all_interactions_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_ALL_DATA_DIR, training_pipeline.DATA_INGESTION_INTERACTIONS_FILE_NAME
        )
        self.all_courses_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_ALL_DATA_DIR, training_pipeline.DATA_INGESTION_COURSES_FILE_NAME
        )
        
        self.interactions_train_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_SPLIT_DATA_DIR, training_pipeline.DATA_INGESTION_INTERACTIONS_TRAIN_FILE_NAME
        )
        self.interactions_test_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_SPLIT_DATA_DIR, training_pipeline.DATA_INGESTION_INTERACTIONS_TEST_FILE_NAME
        )
        self.interactions_split_percentage: float = training_pipeline.DATA_INGESTION_INTERACTIONS_SPLIT_PERCENTAGE
        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_courses_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_TRAINER_TRAINED_COURSES_MODEL_NAME
        )
        self.trained_interactions_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_TRAINER_TRAINED_INTERACTIONS_MODEL_NAME
        )
        

class ModelEvaluationConfig: 
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = os.path.join(self.model_evaluation_dir,training_pipeline.MODEL_EVALUATION_REPORT_NAME)
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE