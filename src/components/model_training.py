
import pandas as pd
import scipy
from src.logger import logging
from src.exception import TrainException
import os,sys
from src.components.data_ingestion import DataIngestion
import numpy as np
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import save_object,write_json_file
from lightfm import LightFM
from lightfm.data import Dataset
class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,\
        data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("MODEL TRAINER:Getting Model Training Configurations")
            self.model_trainer_config = model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact

            logging.info("MODEL TRAINER:Model Training Configured")            
        except Exception as e:
            raise TrainException(e,sys)
        

    def _feature_colon_value(self,value_list):
        """
        Takes as input a list and prepends the columns names to respective values in the list.
        For example: if value_list = [9,7,5.2,'cat'], feature columns are = [f1,f2,f3,f4]
        resultant output = ['f1:9', 'f2:7', 'f3:5.2', 'f4:cat']
        """
        try:
            logging.info("MODEL TRAINER: Prepending the columns names to respective values")
            result = []
            feature_columns = ['user_id:', 'prev_web_dev:', 'prev_data_sc:', 'prev_data_an:', 'prev_game_dev:', 'prev_mob_dev:',\
                            'prev_program:', 'prev_cloud:', 'yrs_of_exp:', 'no_certifications:']
            aa = value_list
            for x,y in zip(feature_columns,aa):
                res = str(x) +""+ str(y)
                result.append(res)
            return result
        
        except Exception as e:
            raise TrainException(e,sys)
           

    def model_training_similar_users(self, train_interactions_file_path, users_all_data_file_path):
        try:
            logging.info("MODEL TRAINER:Into the model_training_similar_users function of ModelTrainer class")

            #read interactions data
            interactionsdf = pd.read_parquet(train_interactions_file_path)

            #read users data
            users = pd.read_parquet(users_all_data_file_path)
            
            logging.info("MODEL TRAINER:Creating LightFM Skeleton Dataset Format")
            #Formatting User Features in the form of [feature_name_1:value_1, feature_name_1:value_2,..., feature_name_n:value_n]
            #This is how lightfm expects the user features for fitting in the skeleton dataset
            user_features = []
            col = ['user_id']*len(users.user_id.unique()) + ['prev_web_dev']*len(users.prev_web_dev.unique()) + ['prev_data_sc']*len(users.prev_data_sc.unique()) + ['prev_data_an']*len(users['prev_data_an'].unique()) \
                + ['prev_game_dev']*len(users.prev_game_dev.unique()) + ['prev_mob_dev']*len(users.prev_mob_dev.unique()) + ['prev_program']*len(users.prev_program.unique()) + ['prev_cloud']*len(users.prev_cloud.unique()) \
                + ['yrs_of_exp']*len(users.yrs_of_exp.unique()) + ['no_certifications']*len(users.no_certifications.unique())

            unique_f1 = list(users.user_id.unique()) + list(users.prev_web_dev.unique()) + list(users.prev_data_sc.unique()) + list(users.prev_data_an.unique()) \
                + list(users.prev_game_dev.unique()) + list(users.prev_mob_dev.unique()) + list(users.prev_program.unique()) + list(users.prev_cloud.unique()) \
                + list(users.yrs_of_exp.unique()) + list(users.no_certifications.unique())

            for x,y in zip(col, unique_f1):
                res = str(x)+ ":" +str(y)
                user_features.append(res)
            
            # We need to call the fit method to tell LightFM who the users are, 
            # what courses we are dealing with, in addition to any user/course features.
            dataset1 = Dataset()
            dataset1.fit(
                interactionsdf['user_id'].unique(), # all the users
                interactionsdf['course_id'].unique(), # all the items
                user_features = user_features
                )
            logging.info("MODEL TRAINER:LightFM Skeleton Dataset Format Created")

            logging.info("MODEL TRAINER:LightFM Building Event-Interactions Sparse Matrices")
            # plugging in the interactions data to build the sparse matrix
            (interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in interactionsdf.values])
            logging.info("MODEL TRAINER:LightFM Event-Interactions Sparse Matrices Built")

            logging.info("MODEL TRAINER:LightFM Building User Features")
            # Formatting User Features in the form of [feature_name_1:value_1, feature_name_1:value_2,..., feature_name_n:value_n]
            # To the Form of ( user_1: [feature_name_1:value_1, feature_name_1:value_2,..., feature_name_n:value_n],
            #                  user_2: [feature_name_1:value_1, feature_name_1:value_2,..., feature_name_n:value_n])
            #This is format to build user features by LightFM
            ad_subset = users[['user_id', 'prev_web_dev', 'prev_data_sc', 'prev_data_an', 'prev_game_dev', 'prev_mob_dev',\
                                  'prev_program','prev_cloud','yrs_of_exp','no_certifications']] 
            ad_list = [list(x) for x in ad_subset.values]
            feature_list = []
            for item in ad_list:
                feature_list.append(self._feature_colon_value(item))

            user_tuple = list(zip(users.user_id, feature_list))
            user_features = dataset1.build_user_features(user_tuple, normalize= False)
            logging.info("MODEL TRAINER:LightFM User Features Built")

            user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
            
            logging.info("MODEL TRAINER:LightFM Model Trainer Created and Model Training Started")
            model = LightFM(no_components=10,loss='warp')
            model.fit(interactions, # spase matrix representing whether user u and course i interacted
                      user_features= user_features, # we have built the sparse matrix above
                      sample_weight= weights, # spase matrix representing how much value to give to user u and course i interaction
                      epochs=20)
            logging.info("MODEL TRAINER:LightFM Model Training Complete")

            n_users, n_items = interactions.shape
            user_courses_number = {'n_users': str(n_users), 'n_items': str(n_items)}
   
            logging.info("Exiting the model_training_similar_users function of ModelTrainer class")
            return model,user_courses_number,user_id_map,user_feature_map
        except Exception as e:
            raise TrainException(e,sys)
        
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("MODEL TRAINER:Into the initiate_model_trainer function of ModelTrainer class")
            train_interactions_file_path = self.data_ingestion_artifact.trained_interactions_file_path

            user_features_file_path = self.data_ingestion_artifact.users_all_data_file_path
            
            logging.info("Training the similar users data")
            model_interactions_train, model_users_courses_number_file_all, model_user_id_map_all, model_user_feature_map_all = self.model_training_similar_users(train_interactions_file_path,user_features_file_path)
            
            logging.info("Saving the similar users model")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_interactions_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.trained_interactions_model_file_path,model_interactions_train)

            logging.info("Saving the users-courses numbers json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.interactions_matrix_shape_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            write_json_file(self.model_trainer_config.interactions_matrix_shape_file_path, model_users_courses_number_file_all)

            logging.info("Saving the users-id-map json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.model_users_id_map_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.model_users_id_map_file_path, model_user_id_map_all)

            logging.info("Saving the users-feature-map json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.model_users_feature_map_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.model_users_feature_map_file_path, model_user_feature_map_all)
            
            logging.info("Saving Model Trainer Artifact")
            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_interactions_model_file_path=self.model_trainer_config.trained_interactions_model_file_path,
                interactions_matrix_shape_file_path = self.model_trainer_config.interactions_matrix_shape_file_path,
                users_id_map_file_path = self.model_trainer_config.model_users_id_map_file_path,
                users_feature_map_file_path = self.model_trainer_config.model_users_feature_map_file_path,
                )

            logging.info(f"MODEL TRAINER:Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise TrainException(e,sys)
    

    