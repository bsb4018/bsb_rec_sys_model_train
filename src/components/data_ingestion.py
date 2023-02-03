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
        except Exception as e:
            raise TrainException(e,sys)

    def get_interaction_features_from_feature_store(self):
        try:
            logging.info("Into the get_interaction_features_from_feature_store function of DataIngestion class")
            store = FeatureStore(repo_path="feature_repo")
            #interaction_df = pd.read_parquet(path = "data-entity/data-interactions-entity.parquet")

            interaction_entity_sql = f"""
                SELECT interaction_id,event_timestamp 
                FROM {store.get_data_source("rs_source_interactions").get_table_query_string()}
                WHERE event_timestamp BETWEEN '2019-01-01' and '2023-01-31'
            """

            logging.info("Getting Interactions Features from Feast")
            interaction_data = store.get_historical_features(entity_df = interaction_entity_sql, features = \
                ["interaction_features:user_id",\
                    "interaction_features:course_id",\
                        "interaction_features:event"]).to_df()
        
            logging.info("Forming the response")
            response_data = interaction_data[["user_id", "course_id", "event"]]
            response_data["user_id"] = response_data["user_id"].astype('int64')
            response_data["user_id"] = response_data["course_id"].astype('int64')
            response_data["event"] = response_data["event"].astype('int64')
            response_data.sort_values(by=["user_id"])
            #Save to the proper directory
            dir_path = os.path.dirname(self.data_ingestion_config.all_interactions_file_path)
            os.makedirs(dir_path, exist_ok=True)
            response_data.to_parquet(self.data_ingestion_config.all_interactions_file_path, index=False)

        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)

    def split_ingested_interaction_features_data(self):
        try:
            # split the data into train and test data
            interactions = pd.read_parquet(self.data_ingestion_config.all_interactions_file_path)

            split_point = int(np.round(interactions.shape[0]*0.7))
            interactions_train = interactions.iloc[0:split_point]
            interactions_valid = interactions.iloc[split_point::]
            #check that user_id and course_id already exist on the train data
            interactions_valid = interactions_valid[(interactions_valid['user_id'].isin(interactions_valid['user_id'])) 
                          & (interactions_valid['course_id'].isin(interactions_valid['course_id']))]

            #Save to the proper directory -> train
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            interactions_train.to_parquet(self.data_ingestion_config.interactions_train_file_path, index=False)

            #Save to the proper directory -> test
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_test_file_path)
            os.makedirs(dir_path, exist_ok=True)
            interactions_valid.to_parquet(self.data_ingestion_config.interactions_test_file_path, index=False)
        except Exception as e:
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


