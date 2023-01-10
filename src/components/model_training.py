
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
from src.utils.main_utils import save_object,save_numpy_array_data

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
            interactions_data_csr = csr_matrix((interactionsdf.rating, (interactionsdf.user_id , interactionsdf.course_id)))
            model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0, iterations=15)
            model.fit(interactions_data_csr)
            logging.info("Exiting the model_training_similar_users function of ModelTrainer class")
            return model
        except Exception as e:
            raise TrainException(e,sys)

    def model_training_similar_courses(self, train_courses_file_path):
        try:
            logging.info("Into the  model_training_similar_courses function of ModelTrainer class")
            coursesdf = pd.read_parquet(train_courses_file_path)
            coursesdf['course_tags'] = coursesdf['course_tags'].str.lower()

            #Train the model on simmilar courses
            #Define a TF-IDF Vectorizer Object. Remove all english stopwords
            tfidf = TfidfVectorizer(stop_words='english')  
            tfidf_matrix = tfidf.fit_transform([str(val) for val in coursesdf["course_tags"] if val is not np.nan])
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            logging.info("Exiting the model_training_similar_courses function of ModelTrainer class")
            return cosine_sim

        except Exception as e:
            raise TrainException(e,sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Into the initiate_model_trainer function of ModelTrainer class")
            train_interactions_file_path = self.data_ingestion_artifact.trained_interactions_file_path
            #test_interactions_file_path = self.data_ingestion_artifact.test_interactions_file_path
            train_courses_file_path = self.data_ingestion_artifact.trained_courses_file_path
   
            logging.info("Training the similar courses data")
            model_courses_train = self.model_training_similar_courses(train_courses_file_path)

            logging.info("Training the similar users data")
            model_interactions_train = self.model_training_similar_users(train_interactions_file_path)
            
            logging.info("Saving the similar courses model")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_courses_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_numpy_array_data(self.model_trainer_config.trained_courses_model_file_path, array=model_courses_train)

            logging.info("Saving the similar users model")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_interactions_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.trained_interactions_model_file_path,model_interactions_train)

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_courses_model_file_path=self.model_trainer_config.trained_courses_model_file_path,
                trained_interactions_model_file_path=self.model_trainer_config.trained_interactions_model_file_path)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise TrainException(e,sys)
    

    