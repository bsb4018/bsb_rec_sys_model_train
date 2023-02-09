
import pandas as pd
import scipy
from src.logger import logging
from src.exception import TrainException
import os,sys
from src.components.data_ingestion import DataIngestion
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import save_object,write_json_file,save_numpy_array_data,save_npz_object
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score

class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,\
        data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact            
        except Exception as e:
            raise TrainException(e,sys)

    def model_training_similar_users(self, train_interactions_file_path):
        try:
            logging.info("Into the model_training_similar_users function of ModelTrainer class")
            #read interactions data
            interactionsdf = pd.read_parquet(train_interactions_file_path)
            id_cols=['user_id','course_id']
            interactions_train_data = dict()
            for k in id_cols:
                interactions_train_data[k] = interactionsdf[k].values
            
            events = dict()
            events['train'] = interactionsdf.event

            n_users=len(np.unique(interactions_train_data['user_id']))
            n_items=len(np.unique(interactions_train_data['course_id']))

            train_matrix = dict()
            train_matrix['train'] = coo_matrix((events['train'], 
                                   (interactions_train_data['user_id'], 
                                    interactions_train_data['course_id'])), 
                                    shape=(n_users,n_items))
            
            user_courses_number = {'n_users': str(n_users), 'n_items': str(n_items)}
            # model creation and training
            model = LightFM(no_components=10, loss='warp')
            model.fit(train_matrix['train'], epochs=100, num_threads=8)

            logging.info("Exiting the model_training_similar_users function of ModelTrainer class")
            return model,user_courses_number
        except Exception as e:
            raise TrainException(e,sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Into the initiate_model_trainer function of ModelTrainer class")
            train_interactions_file_path = self.data_ingestion_artifact.trained_interactions_file_path
            
            logging.info("Training the similar users data")
            model_interactions_train, model_users_courses_number_file = self.model_training_similar_users(train_interactions_file_path)
            
            logging.info("Saving the similar users model")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_interactions_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.trained_interactions_model_file_path,model_interactions_train)

            logging.info("Saving the number of users-courses json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.interactions_matrix_shape_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            write_json_file(self.model_trainer_config.interactions_matrix_shape_file_path, model_users_courses_number_file)
            #save_object(self.model_trainer_config.interactions_matrix_file_path, interactions_data_csr)
            #save_npz_object(self.model_trainer_config.interactions_matrix_file_path, interactions_data_csr)
            #scipy.sparse.save_npz(model_dir_path, interactions_data_csr)

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_interactions_model_file_path=self.model_trainer_config.trained_interactions_model_file_path,
                interactions_matrix_shape_file_path = self.model_trainer_config.interactions_matrix_shape_file_path
                )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise TrainException(e,sys)
    

    