from src.logger import logging
from src.exception import TrainException
import os,sys
import pandas as pd
import numpy as np
from src.ml.model.model_recommendation import Recommendation
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object, write_json_file, read_json_file
from scipy.sparse import csr_matrix
import json, yaml
import math
from pathlib import Path
from src.ml.model.model_resolver import ModelResolver

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                       data_ingestion_artifact: DataIngestionArtifact,
                       model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_recommendation = Recommendation()
            
        except Exception as e:
            raise TrainException(e,sys)

    def model_evaluating_similar_users(self):
        try:

            #Load train data
            interactions_train_data = pd.read_parquet(self.data_ingestion_artifact.trained_interactions_file_path)

            #Load test data
            interactions_test_data = pd.read_parquet(self.data_ingestion_artifact.test_interactions_file_path)

            #Load trained model
            model = load_object(self.model_trainer_artifact.trained_interactions_model_file_path)

            interactions_test_csr = csr_matrix((interactions_test_data.rating, (interactions_test_data.user_id , interactions_test_data.course_id)))

            generated_recommendations = self.model_recommendation.get_recommendations_similar_users_all(interactions_test_csr,model)

            valid_user_courses = interactions_test_data.groupby('user_id')['course_id'].apply(set).to_dict()

            train_users = np.sort(interactions_train_data.user_id.unique())
            valid_users = np.sort(interactions_test_data.user_id.unique())
            combined_users = set(train_users) & set(valid_users)

        
            index_to_user = pd.Series([0])
            index_to_item = pd.Series([0])
            index_to_user1 = pd.Series(np.sort(np.unique(interactions_train_data['user_id'])))
            index_to_item1 = pd.Series(np.sort(np.unique(interactions_train_data['course_id'])))
            index_to_user = index_to_user.append(index_to_user1,ignore_index=True)
            index_to_item = index_to_item.append(index_to_item1,ignore_index=True)
            generated_recommendations = pd.DataFrame(generated_recommendations, index=index_to_user.values).apply(\
                lambda c: c.map(index_to_item))

            model_hitrate = np.mean([int(len(set(generated_recommendations.loc[u]) & valid_user_courses[u]) > 0) for u in combined_users])
            model_preison = np.mean([len(set(generated_recommendations.loc[u]) & valid_user_courses[u]) / len(generated_recommendations.loc[u]) for u in combined_users])
            model_recall = np.mean([len(set(generated_recommendations.loc[u]) & valid_user_courses[u]) / len(valid_user_courses[u]) for u in combined_users])

            #create a dictionary of the merrics and save it
            evaluation_report = {'hit_rate': model_hitrate*100, 'precision': model_preison*100, 'recall': model_recall*100}
            write_json_file(self.model_eval_config.report_file_path, evaluation_report)
      

            is_model_accepted = True
            model_resolver = ModelResolver()
                       
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_hitrate=None,
                    best_model_path=None,
                    best_model_report_path=None,
                    current_trained_model_path=self.model_trainer_artifact.trained_interactions_model_file_path,
                    current_model_report_file_path=self.model_eval_config.report_file_path
                )
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)

            best_model_report_path = model_resolver.get_best_model_report_path()
            best_model_report = read_json_file(best_model_report_path)
            best_hit_rate = best_model_report["hit_rate"] / 100
            current_hit_rate = model_hitrate

            improved_hitrate = abs(math.floor(current_hit_rate) - math.floor(best_hit_rate))
            if improved_hitrate >= self.model_eval_config.change_threshold:
                is_model_accepted=True   
            else:
                is_model_accepted=False
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = is_model_accepted,
                improved_hitrate = improved_hitrate,
                best_model_path = latest_model_path,
                best_model_report_path = best_model_report_path,
                current_trained_model_path = self.model_trainer_artifact.trained_interactions_model_file_path,
                current_model_report_file_path = self.model_eval_config.report_file_path)
           
            return model_evaluation_artifact    

        except Exception as e:
            raise TrainException(e,sys)

