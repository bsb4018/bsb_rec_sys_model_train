import os

SAVED_MODEL_DIR =os.path.join("saved_models")
PIPELINE_NAME: str = "course_recommend"
ARTIFACT_DIR: str = "artifact"

'''
Defining basic and common file names
'''

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

