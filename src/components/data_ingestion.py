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
from src.configurations.aws_config import StorageConnection

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            logging.info("DATA INGESTION:Getting Data Ingestion Configurations")
            self.connection = StorageConnection()
            self.data_ingestion_config = data_ingestion_config

            logging.info("DATA INGESTION:Data Ingestion Configured")
        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)
    
    def get_feature_registry_and_data_from_s3(self):
        try:
            logging.info("DATA INGESTION:Downloading Feature Store Files and Registries")
            self.connection.download_feature_store_registries_s3()

            logging.info("DATA INGESTION:Downloaded Feature Store Files and Registries")
        except Exception as e:
            raise TrainException(e,sys)
        
           
    def get_interaction_features_from_feature_store(self):
        try:
            logging.info("DATA INGESTION:Into the get_interaction_features_from_feature_store function of DataIngestion class")
            store = FeatureStore(repo_path="feature_repo")

            #Defining the interaction features to bring by mentioning a timeframe in an sql query
            interaction_entity_sql = f"""
                SELECT event_timestamp,interaction_id
                FROM {store.get_data_source("rs_source_interactions").get_table_query_string()}
                WHERE event_timestamp BETWEEN '2019-01-01' and '2023-02-11'
            """

            logging.info("Getting Interactions Features from Feast")
            interaction_data = store.get_historical_features(entity_df = interaction_entity_sql, features = \
                ["interaction_features:user_id",\
                    "interaction_features:course_id",\
                        "interaction_features:event"]).to_df()
        
            logging.info("Forming the required features dataframe from the response given by feature store")
            response_data = interaction_data[["user_id", "course_id", "event"]]
            #Save to the proper directory
            dir_path = os.path.dirname(self.data_ingestion_config.all_interactions_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Interaction Features Ingested")
            response_data.to_parquet(self.data_ingestion_config.all_interactions_file_path, index=False)

            logging.info("DATA INGESTION:Exiting get_interaction_features_from_feature_store function of DataIngestion class")

        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)
        
    def get_user_features_from_feature_store(self):
        try:
            logging.info("DATA INGESTION:Into the get_user_features_from_feature_store function of DataIngestion class")
            store = FeatureStore(repo_path="feature_repo")
            
            #Defining the user features to bring by mentioning a timeframe in an sql query
            user_entity_sql = f"""
                SELECT event_timestamp,user_feature_id
                FROM {store.get_data_source("rs_source_users").get_table_query_string()} 
                WHERE event_timestamp BETWEEN '2019-01-01' and '2023-02-11'
            """

            logging.info("Getting Interactions Features from Feast")
            #logging.info("Getting Features from Feast")
            users_data = store.get_historical_features(entity_df = user_entity_sql, \
            features = ["user_features:prev_web_dev","user_features:prev_data_sc","user_features:prev_data_an",\
                        "user_features:prev_game_dev","user_features:prev_mob_dev","user_features:prev_program",\
                            "user_features:prev_cloud","user_features:yrs_of_exp","user_features:no_certifications",\
                                "user_features:user_id"]).to_df()
            
            logging.info("Forming the required features dataframe from the response given by feature store")
            response_data = users_data[["user_id","prev_web_dev","prev_data_sc","prev_data_an",\
                        "prev_game_dev","prev_mob_dev","prev_program",\
                            "prev_cloud","yrs_of_exp","no_certifications"]]
                   
            #Save to the proper directory
            dir_path = os.path.dirname(self.data_ingestion_config.all_users_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("User Features Ingested")
            response_data.to_parquet(self.data_ingestion_config.all_users_file_path, index=False)

            logging.info("DATA INGESTION:Exiting get_user_features_from_feature_store function of DataIngestion class")

        except Exception as e:
            logging.exception(e)
            raise TrainException(e,sys)
    

    def split_ingested_interaction_features_data(self):
        try:
            logging.info("DATA INGESTION:Into the split_ingested_interaction_features_data function of DataIngestion class")
            # split the data into train and test data
            interactions = pd.read_parquet(self.data_ingestion_config.all_interactions_file_path)
            
            #Splitting the data into training:testing as 80:20
            split_point = int(np.round(interactions.shape[0]*0.8))
            interactions_train = interactions.iloc[0:split_point]
            interactions_valid = interactions.iloc[split_point::]

            #check that all the user_id and course_id from validation data already exist on the train data
            interactions_valid = interactions_valid[(interactions_valid['user_id'].isin(interactions_train['user_id'])) 
                          & (interactions_valid['course_id'].isin(interactions_train['course_id']))]
            
            #split the train users features accordingly
            #users = pd.read_parquet(self.data_ingestion_config.all_users_file_path)
            #users = users[(users['user_id'].isin(interactions_train['user_id']))]

            #Save to the proper directory -> train
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            interactions_train.to_parquet(self.data_ingestion_config.interactions_train_file_path, index=False)

            #Save to the proper directory -> test
            dir_path = os.path.dirname(self.data_ingestion_config.interactions_test_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Interactions Data Splits Saved")
            interactions_valid.to_parquet(self.data_ingestion_config.interactions_test_file_path, index=False)

            logging.info("DATA INGESTION:Exiting split_ingested_interaction_features_data function of DataIngestion class")

        except Exception as e:
            raise TrainException(e,sys)
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("DATA INGESTION:Entered initiate_data_ingestion method of Data_Ingestion class")
 
            self.get_feature_registry_and_data_from_s3()

            self.get_user_features_from_feature_store()

            self.get_interaction_features_from_feature_store()
            
            self.split_ingested_interaction_features_data()

            logging.info("DATA INGESTION:Saving the Data Ingestion Artifact")

            data_ingestion_artifact = DataIngestionArtifact( 
                trained_interactions_file_path = self.data_ingestion_config.interactions_train_file_path,
                test_interactions_file_path = self.data_ingestion_config.interactions_test_file_path,
                interactions_all_data_file_path = self.data_ingestion_config.all_interactions_file_path,
                users_all_data_file_path = self.data_ingestion_config.all_users_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            logging.info("DATA INGESTION: Successful")
            return data_ingestion_artifact
        except Exception as e:
            raise TrainException(e, sys) from e


