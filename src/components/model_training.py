'''
import pandas as pd
import scipy
from src.logger import logging
from src.exception import TrainException
import os,sys
from feast import FeatureStore
from src.components.data_ingestion import DataIngestion
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class ModelTrainEval:
    def __init__(self):
        try:
            self.data_ingestion = DataIngestion()
            self.data_split = DataSplit()
        except Exception as e:
            raise TrainException(e,sys)

    
    def model_training_similar_users(self):
        try:
            interactionsdf = self.split_ingested_interaction_features_data(mode="train")
            interactions_data_csr = csr_matrix((interactionsdf.rating, (interactionsdf.user_id , interactionsdf.course_id)))
            model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
            model.fit(interactions_data_csr)
            #save the model
        except Exception as e:
            raise TrainException(e,sys)

    def model_training_similar_courses(self):
        try:
            #interactionsdf = self.load_interactions_ingested_features()
            coursesdf = self.data_ingestion.get_course_features_from_feature_store()
            coursesdf['course_tags'] = coursesdf['course_tags'].str.lower()

            #Train the model on simmilar courses
            #Define a TF-IDF Vectorizer Object. Remove all english stopwords
            tfidf = TfidfVectorizer(stop_words='english')  
            tfidf_matrix = tfidf.fit_transform([str(val) for val in coursesdf["course_tags"] if val is not np.nan])
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

            #save the model 
            outfile = "from model training artifact"
            np.save(outfile, cosine_sim)
            return cosine_sim

        except Exception as e:
            raise TrainException(e,sys)
'''
    

    

    