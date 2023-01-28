import pandas as pd
import scipy
from src.logger import logging
from src.exception import TrainException
import os,sys
import numpy as np
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
#from src.constants.file_path_constants import FEAST_FEATURE_STORE_REPO_PATH,INTERACTIONS_DATA_FILE_PATH,COURSES_DATA_FILE_PATH
from sklearn.model_selection import train_test_split
from src.configurations.aws_config import StorageConnection

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.connection = StorageConnection()
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)
    
    def get_feature_registry_and_data_from_s3(self):
        try:
            self.connection.download_feature_store_registries_s3()
            self.connection.download_entity_data_s3()
        except Exception as e:
            raise TrainException(e,sys)

    def get_interaction_features_from_feature_store(self):
        try:
            logging.info("Into the get_interaction_features_from_feature_store function of DataIngestion class")
            store = FeatureStore(repo_path="D:/work2/course_recommend_app/cr_data_collection/rec_sys_fs")
            interaction_df1 = pd.read_parquet(path = "data/interactions.parquet")

            logging.info("Getting Interactions Features from Feast")
            interaction_data = store.get_historical_features(entity_df = interaction_df1, features = \
                ["interactions_df_feature_view:user_id",\
                    "interactions_df_feature_view:course_id",\
                        "interactions_df_feature_view:rating"]).to_df()
        
            logging.info("Forming the response")
            response_data = interaction_data[["user_id", "course_id", "rating"]]

            #Save to the proper directory
            dir_path = os.path.dirname(self.data_ingestion_config.all_interactions_file_path)
            os.makedirs(dir_path, exist_ok=True)
            response_data.to_parquet(self.data_ingestion_config.all_interactions_file_path, index=False)

        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)
    
    def split_ingested_interaction_features_data(self):
        try:
            logging.info("Entered split_ingested_interaction_features_data method of Data_Ingestion class")
            interactions = pd.read_parquet(self.data_ingestion_config.all_interactions_file_path)

            test_size = self.data_ingestion_config.interactions_split_percentage
            interactions_train, interactions_valid = train_test_split(interactions, test_size=test_size, random_state=48)
            interactions_train = interactions_train.groupby(['user_id', 'course_id']).size().to_frame('rating').reset_index()
            interactions_valid = interactions_valid.groupby(['user_id', 'course_id']).size().to_frame('rating').reset_index()
            
            '''
            valid_data_percentage = self.data_ingestion_config.interactions_split_percentage
            interactions['random'] = np.random.random(size=len(interactions))
            train_mask = interactions['random'] <  (1 - valid_data_percentage)
            valid_mask = interactions['random'] >= (1 - valid_data_percentage)

            #Ordering the data
            interactions_train = interactions[train_mask].groupby(['user_id', 'course_id']).size().to_frame('rating').reset_index()
            interactions_valid = interactions[valid_mask].groupby(['user_id', 'course_id']).size().to_frame('rating').reset_index()
            '''
            sample_weight_train = np.log2(interactions_train['rating'] + 1)
            sample_weight_valid = np.log2(interactions_valid['rating'] + 1)
        
            #interactions_train = interactions_train[['user_id', 'course_id']]
            #interactions_valid = interactions_valid[['user_id', 'course_id']]

            #train_users = np.sort(interactions_train.user_id.unique())
            #valid_users = np.sort(interactions_valid.user_id.unique())
            #cold_start_users = set(valid_users) - set(train_users)

            #train_items = np.sort(interactions_train.course_id.unique())
            #valid_items = np.sort(interactions_valid.course_id.unique())
            #cold_start_items = set(valid_items) - set(train_items)

            interactions_train['rating'] = sample_weight_train
            interactions_valid['rating'] = sample_weight_valid

            #item_features_train = courses_features_only[courses_features_only.course_id.isin(train_items)]
            #item_features_valid = courses_features_only[courses_features_only.course_id.isin(valid_items)]

            #Save to the proper directory -> train
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            interactions_train.to_parquet(self.data_ingestion_config.interactions_train_file_path, index=False)

            #Save to the proper directory -> test
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_test_file_path)
            os.makedirs(dir_path, exist_ok=True)
            interactions_valid.to_parquet(self.data_ingestion_config.interactions_test_file_path, index=False)
            
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

            self.get_feature_registry_and_data_from_s3()

            self.get_interaction_features_from_feature_store()
            
            self.split_ingested_interaction_features_data()

            data_ingestion_artifact = DataIngestionArtifact( 
                trained_interactions_file_path = self.data_ingestion_config.interactions_train_file_path,
                test_interactions_file_path = self.data_ingestion_config.interactions_test_file_path,
                interactions_all_data_file_path = self.data_ingestion_config.all_interactions_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise TrainException(e, sys) from e


