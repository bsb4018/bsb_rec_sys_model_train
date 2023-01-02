import pandas as pd
import scipy
from src.logger import logging
from src.exception import TrainException
import os,sys
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from src.constants.file_path_constants import FEAST_FEATURE_STORE_REPO_PATH,INTERACTIONS_DATA_FILE_PATH,COURSES_DATA_FILE_PATH,USERS_DATA_FILE_PATH
class DataIngestion:
    def __init__(self):
        try:
            self.store = FeatureStore(repo_path=FEAST_FEATURE_STORE_REPO_PATH)
            self.interaction_df = pd.read_parquet(path = INTERACTIONS_DATA_FILE_PATH)
            self.courses_df = pd.read_parquet(path = COURSES_DATA_FILE_PATH)
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)

    def get_course_features_from_feature_store(self):
        try:
            logging.info("Into the get_course_features_from_feature_store function of DataIngestion class")
            courses_df1 = self.courses_df

            logging.info("Getting Course Features from Feast")
            courses_data = self.store.get_historical_features(entity_df = courses_df1, features = \
                ["courses_df_feature_view:course_id",\
                    "courses_df_feature_view:course_name", "courses_df_feature_view:course_tags"]).to_df()
        
            logging.info("Forming the response")
            response_data = courses_data[["course_id", "course_name", "course_tags"]]

            return response_data

        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)

    def get_users_features_from_feature_store(self):
        try:
            pass
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)

    def get_interaction_features_from_feature_store(self):
        try:
            logging.info("Into the get_interaction_features_from_feature_store function of DataIngestion class")
            interaction_df1 = self.interaction_df

            logging.info("Getting Interactions Features from Feast")
            interaction_data = self.store.get_historical_features(entity_df = interaction_df1, features = \
                ["interactions_df_feature_view:user_id",\
                    "interactions_df_feature_view:course_id",\
                        "interactions_df_feature_view:rating"]).to_df()
        
            logging.info("Forming the response")
            response_data = interaction_data[["user_id", "course_id", "rating"]]

            return response_data
            
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)


