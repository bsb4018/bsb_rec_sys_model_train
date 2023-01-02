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

class ModelTrainer:
    def __init__(self):
        try:
            self.data_ingestion = DataIngestion()
        except Exception as e:
            raise TrainException(e,sys)

    def load_interactions_ingested_features(self):
        try:
            interactionsdf = self.data_ingestion.get_interaction_features_from_feature_store()
            return interactionsdf
        except Exception as e:
            raise TrainException(e,sys)

    def load_courses_ingested_features(self):
        try:
            coursesdf = self.data_ingestion.get_course_features_from_feature_store()
            return coursesdf
        except Exception as e:
            raise TrainException(e,sys)
    
    def model_training_similar_users(self):
        try:
            interactionsdf = self.load_interactions_ingested_features()
            interactions_data_csr = csr_matrix((interactionsdf.rating, (interactionsdf.user_id , interactionsdf.course_id)))
            model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
            model.fit(interactions_data_csr)
            #save the model
        except Exception as e:
            raise TrainException(e,sys)

    def model_training_similar_courses(self):
        try:
            #interactionsdf = self.load_interactions_ingested_features()
            coursesdf = self.load_courses_ingested_features()
            coursesdf['course_tags'] = coursesdf['course_tags'].str.lower()

            #Train the model on simmilar courses
            #Define a TF-IDF Vectorizer Object. Remove all english stopwords
            tfidf = TfidfVectorizer(stop_words='english')  
            tfidf_matrix = tfidf.fit_transform([str(val) for val in coursesdf["course_tags"] if val is not np.nan])
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            #save the model

            return cosine_sim
        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations(self, course_name: str):
        try:
            coursesdf = self.load_courses_ingested_features()
            cosine_sim = self.model_training_similar_courses()

            #Construct a reverse mapping of indices and course names, and drop duplicate titles, if any
            indices = pd.Series(coursesdf.index,index=coursesdf['course_name']).drop_duplicates()

            idx = indices[course_name]
            found = False
            # Get the pairwsie similarity scores of all courses with that course
            # And convert it into a list of tuples as described above
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Sort the courses based on the cosine similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Get the scores of the 5-6 most similar courses. Ignore the first courses.
            sim_scores = sim_scores[1:6]
            # Get the courses indices
            course_indices = [i[0] for i in sim_scores]
            for cidx in course_indices:
                if indices[course_name] == cidx:
                    found == True
                    course_indices.remove(cidx)
                    break

            if found == False:
                course_indices.pop()

            # Return the top 5-6 most similar courses
            recommended_courses =  coursesdf['course_name'].iloc[course_indices].tolist()
            return recommended_courses

        except Exception as e:
            raise TrainException(e,sys)

    