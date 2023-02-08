from src.logger import logging
from src.exception import TrainException
import os,sys
import pandas as pd
import numpy as np
from src.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object, write_json_file, read_json_file
from scipy.sparse import csr_matrix
import json, yaml
import math
from pathlib import Path
from src.ml.model.model_resolver import ModelResolver
import shutil
from scipy.sparse import coo_matrix
from lightfm.evaluation import auc_score
class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                       data_ingestion_artifact: DataIngestionArtifact,
                       model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            
        except Exception as e:
            raise TrainException(e,sys)


    def model_evaluating_similar_users(self):
        try:

            #Load test data
            interactions_test_df = pd.read_parquet(self.data_ingestion_artifact.test_interactions_file_path)

            #Load trained model
            model = load_object(self.model_trainer_artifact.trained_interactions_model_file_path)

            id_cols=['user_id','course_id']
            interactions_test_data = dict()
            for k in id_cols:
                interactions_test_data[k] = interactions_test_df[k].values
            
            events = dict()
            events['test'] = interactions_test_df.event

            n_users=len(np.unique(interactions_test_data['user_id']))
            n_items=len(np.unique(interactions_test_data['course_id']))

            test_matrix = dict()
            test_matrix['test'] = coo_matrix((events['test'], 
                                   (interactions_test_data['user_id'], 
                                    interactions_test_data['course_id'])), 
                                    shape=(n_users,n_items))

            # model creation and training
            mean_auc_score = auc_score(model, test_matrix['test'], num_threads=10).mean()

            #create a dictionary of the merrics and save it
            evaluation_report = {'mean_accuracy_score': str(mean_auc_score)}
            write_json_file(self.model_eval_config.report_file_path, evaluation_report)

            return mean_auc_score
    
        except Exception as e:
            raise TrainException(e,sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:

            current_model_mean_auc_score = self.model_evaluating_similar_users()

            #model_evaluation_path = self.model_eval_config.model_evaluation_dir
            #os.makedirs(os.path.dirname(model_evaluation_path),exist_ok=True)
            #shutil.copy(src=self.model_eval_config.report_file_path, dst=self.model_eval_config.report_file_path)
            self.model_trainer_artifact.trained_interactions_model_file_path
            
            is_model_accepted = True
            model_resolver = ModelResolver()
                       
            if not model_resolver.is_model_exists():

                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_hitrate=None,
                    best_model_path=self.model_trainer_artifact.trained_interactions_model_file_path,
                    best_model_report_path=self.model_eval_config.report_file_path,
                    current_interactions_model_path=self.model_trainer_artifact.trained_interactions_model_file_path,
                    current_interactions_model_report_file_path=self.model_eval_config.report_file_path,
                    interactions_matrix_file_path = self.model_trainer_artifact.interactions_matrix_file_path
                )
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)

            best_model_report_path = model_resolver.get_best_model_report_path()
            best_model_report = read_json_file(best_model_report_path)
            best_hit_rate = float(best_model_report["mean_accuracy_score"])
            current_hit_rate = current_model_mean_auc_score

            improved_hitrate = abs(current_hit_rate - best_hit_rate)
            if improved_hitrate >= self.model_eval_config.change_threshold:
                is_model_accepted=True   
            else:
                is_model_accepted=False
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = is_model_accepted,
                improved_hitrate = improved_hitrate,
                best_model_path = latest_model_path,
                best_model_report_path = best_model_report_path,
                current_interactions_model_path = self.model_trainer_artifact.trained_interactions_model_file_path,
                current_interactions_model_report_file_path=self.model_eval_config.report_file_path,
                interactions_matrix_file_path = self.model_trainer_artifact.interactions_matrix_file_path)
           
            return model_evaluation_artifact
        except Exception as e:
            raise TrainException(e,sys)
