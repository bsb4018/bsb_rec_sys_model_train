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
from lightfm.data import Dataset
from lightfm.evaluation import auc_score,precision_at_k
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
        
    def _feature_colon_value(self,my_list):
        """
        Takes as input a list and prepends the columns names to respective values in the list.
        For example: if my_list = [1,1,0,'del'],
        resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']
        """
        result = []
        ll = ['user:', 'prev_web_dev:', 'prev_data_sc:', 'prev_data_an:', 'prev_game_dev:', 'prev_mob_dev:',\
                        'prev_program:', 'prev_cloud:', 'yrs_of_exp:', 'no_certifications:']
        aa = my_list
        for x,y in zip(ll,aa):
            res = str(x) +""+ str(y)
            result.append(res)
        return result


    def model_evaluating_similar_users(self):
        try:

            #Load test data
            interactions_test_df = pd.read_parquet(self.data_ingestion_artifact.test_interactions_file_path)

            usersdf = pd.read_parquet(self.data_ingestion_artifact.users_all_data_file_path)
            users = usersdf[(usersdf['user_id'].isin(interactions_test_df['user_id']))]

            uf = []
            col = ['user']*len(users.user_id.unique()) + ['prev_web_dev']*len(users.prev_web_dev.unique()) + ['prev_data_sc']*len(users.prev_data_sc.unique()) + ['prev_data_an']*len(users['prev_data_an'].unique()) \
                + ['prev_game_dev']*len(users.prev_game_dev.unique()) + ['prev_mob_dev']*len(users.prev_mob_dev.unique()) + ['prev_program']*len(users.prev_program.unique()) + ['prev_cloud']*len(users.prev_cloud.unique()) \
                + ['yrs_of_exp']*len(users.yrs_of_exp.unique()) + ['no_certifications']*len(users.no_certifications.unique())

            unique_f1 = list(users.user_id.unique()) + list(users.prev_web_dev.unique()) + list(users.prev_data_sc.unique()) + list(users.prev_data_an.unique()) \
                + list(users.prev_game_dev.unique()) + list(users.prev_mob_dev.unique()) + list(users.prev_program.unique()) + list(users.prev_cloud.unique()) \
                + list(users.yrs_of_exp.unique()) + list(users.no_certifications.unique())


            for x,y in zip(col, unique_f1):
                res = str(x)+ ":" +str(y)
                uf.append(res)

            
            # we call fit to supply userid, item id and user/item features
            dataset1 = Dataset()
            dataset1.fit(
                interactions_test_df['user_id'].unique(), # all the users
                interactions_test_df['course_id'].unique(), # all the items
                user_features = uf
                )
            
            # plugging in the interactions and their weights
            (interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in interactions_test_df.values])
            
            ad_subset = users[['user_id', 'prev_web_dev', 'prev_data_sc', 'prev_data_an', 'prev_game_dev', 'prev_mob_dev',\
                                  'prev_program','prev_cloud','yrs_of_exp','no_certifications']] 
            ad_list = [list(x) for x in ad_subset.values]
            feature_list = []
            for item in ad_list:
                feature_list.append(self._feature_colon_value(item))

            user_tuple = list(zip(users.user_id, feature_list))
            user_features = dataset1.build_user_features(user_tuple, normalize= False)
            user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
            n_users, n_items = interactions.shape

            #Load trained model
            model = load_object(self.model_trainer_artifact.trained_interactions_model_file_path)

            # model creation and training
            mean_auc_score = auc_score(model, interactions, user_features=user_features).mean()

            mean_precision_at_10 = precision_at_k(model,
                      interactions,
                      user_features=user_features,
                      k=10
                     ).mean()

            #create a dictionary of the merrics and save it
            evaluation_report = {'mean_accuracy_score': str(mean_auc_score), 'mean_precision_at_10': str(mean_precision_at_10)}
            write_json_file(self.model_eval_config.report_file_path, evaluation_report)

            return mean_precision_at_10
    
        except Exception as e:
            raise TrainException(e,sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:

            current_model_mean_precision_at10 = self.model_evaluating_similar_users()

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
                    interactions_matrix_shape_file_path = self.model_trainer_artifact.interactions_matrix_shape_file_path,
                    current_interactions_model_report_file_path=self.model_eval_config.report_file_path,                    
                )
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)

            best_model_report_path = model_resolver.get_best_model_report_path()
            best_model_report = read_json_file(best_model_report_path)
            best_precision_at_k = float(best_model_report["mean_precision_at_10"])
            current_precision_at_k = current_model_mean_precision_at10

            improved_hitrate = abs(current_precision_at_k - best_precision_at_k)
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
                interactions_matrix_shape_file_path = self.model_trainer_artifact.interactions_matrix_shape_file_path,
                current_interactions_model_report_file_path=self.model_eval_config.report_file_path
                )
           
            return model_evaluation_artifact
        except Exception as e:
            raise TrainException(e,sys)
