
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
from lightfm.data import Dataset
class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,\
        data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact            
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
           

    def model_training_similar_users(self, train_interactions_file_path, users_all_data_file_path):
        try:
            logging.info("Into the model_training_similar_users function of ModelTrainer class")
            #read interactions data
            interactionsdf = pd.read_parquet(train_interactions_file_path)
            #user_ids = interactionsdf.user_id.unique()

            usersdf = pd.read_parquet(users_all_data_file_path)
            users = usersdf[(usersdf['user_id'].isin(interactionsdf['user_id']))]

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
                interactionsdf['user_id'].unique(), # all the users
                interactionsdf['course_id'].unique(), # all the items
                user_features = uf
                )
            
            # plugging in the interactions and their weights
            (interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in interactionsdf.values])
            
            ad_subset = users[['user_id', 'prev_web_dev', 'prev_data_sc', 'prev_data_an', 'prev_game_dev', 'prev_mob_dev',\
                                  'prev_program','prev_cloud','yrs_of_exp','no_certifications']] 
            ad_list = [list(x) for x in ad_subset.values]
            feature_list = []
            for item in ad_list:
                feature_list.append(self._feature_colon_value(item))

            user_tuple = list(zip(users.user_id, feature_list))
            user_features = dataset1.build_user_features(user_tuple, normalize= False)
            user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()

            model = LightFM(no_components=10,loss='warp')
            model.fit(interactions, # spase matrix representing whether user u and item i interacted
                      user_features= user_features, # we have built the sparse matrix above
                      sample_weight= weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings
                      epochs=20)
            
            n_users, n_items = interactions.shape
            user_courses_number = {'n_users': str(n_users), 'n_items': str(n_items)}
            # model creation and training
            #model = LightFM(no_components=10, loss='warp')

            logging.info("Exiting the model_training_similar_users function of ModelTrainer class")
            return model,user_courses_number,user_feature_map
        except Exception as e:
            raise TrainException(e,sys)
        

    '''
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
    '''
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Into the initiate_model_trainer function of ModelTrainer class")
            train_interactions_file_path = self.data_ingestion_artifact.trained_interactions_file_path

            user_features_file_path = self.data_ingestion_artifact.users_all_data_file_path
            
            logging.info("Training the similar users data")
            model_interactions_train, model_users_courses_number_file, model_user_feature_map = self.model_training_similar_users(train_interactions_file_path,user_features_file_path)
            
            logging.info("Saving the similar users model")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_interactions_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.trained_interactions_model_file_path,model_interactions_train)

            logging.info("Saving the number of users-courses json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.interactions_matrix_shape_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            write_json_file(self.model_trainer_config.interactions_matrix_shape_file_path, model_users_courses_number_file)

            logging.info("Saving the users-map json file")
            model_dir_path = os.path.dirname(self.model_trainer_config.model_users_map_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            save_object(self.model_trainer_config.model_users_map_file_path, model_user_feature_map)
            
            #save_object(self.model_trainer_config.interactions_matrix_file_path, interactions_data_csr)
            #save_npz_object(self.model_trainer_config.interactions_matrix_file_path, interactions_data_csr)
            #scipy.sparse.save_npz(model_dir_path, interactions_data_csr)

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_interactions_model_file_path=self.model_trainer_config.trained_interactions_model_file_path,
                interactions_matrix_shape_file_path = self.model_trainer_config.interactions_matrix_shape_file_path,
                users_map_file_path = self.model_trainer_config.model_users_map_file_path
                )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise TrainException(e,sys)
    

    