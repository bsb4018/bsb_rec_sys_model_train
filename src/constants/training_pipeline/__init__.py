import os


'''
Defining basic and common file names
'''

PIPELINE_NAME: str = "course_recommend"
ARTIFACT_DIR: str = "artifact"


'''
Data Ingestion related constant start with DATA_INGESTION VAR NAME
'''

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_ALL_DATA_DIR: str = "all_data"
DATA_INGESTION_SPLIT_DATA_DIR: str = "splitted_data"
DATA_INGESTION_INTERACTIONS_FILE_NAME: str = "interactions.parquet"
DATA_INGESTION_COURSES_FILE_NAME: str = "courses.parquet"
DATA_INGESTION_INTERACTIONS_TRAIN_FILE_NAME: str = "train_interactions.parquet"
DATA_INGESTION_INTERACTIONS_TEST_FILE_NAME: str = "test_interactions.parquet"
DATA_INGESTION_INTERACTIONS_SPLIT_PERCENTAGE: float = 0.25

'''
MODEL TRAINER related constant start with MODEL_TRAINER var name
'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_COURSES_MODEL_NAME: str = "model_courses.pkl"
MODEL_TRAINER_TRAINED_INTERACTIONS_MODEL_NAME: str = "model_interactions.pkl"
MODEL_TRAINER_INTERACTIONS_MATRIX_FILE_NAME: str = "interactions_matrix.pkl"

'''
MODEL EVALUATION ralated constant start with MODEL_EVALUATION VAR NAME
'''

MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_HITRATE_THRESHOLD: float = 1
MODEL_EVALUATION_REPORT_NAME= "report.json"

'''
Model Pusher ralated constant start with MODEL_PUSHER VAR NAME
'''
MODEL_PUSHER_DIR_NAME = "model_pusher"
BEST_INTERACTIONS_MODEL_FILE_NAME = "interactions_best.pkl"
BEST_COURSES_MODEL_FILE_NAME = "courses_best.pkl"
BEST_INTERACTIONS_MODEL_REPORT_FILE_NAME = "best_interactions_model_report.json"


'''
Saved Models for Production
'''
SAVED_MODEL_DIR = os.path.join("saved_models")
PRODUCTION_INTERACTIONS_MODEL_FILE_NAME = "production_interactions_model.pkl"
PRODUCTION_COURSES_MODEL_FILE_NAME = "production_courses_model.pkl"
PRODUCTION_INTERACTIONS_MODEL_REPORT_FILE_NAME = "production_interactions_model_report.json"
PRODUCTION_INTERACTIONS_MATRIX_FILE_NAME = "production_interaction_matrix.npz"