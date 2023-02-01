from src.logger import logging
from src.exception import TrainException
import sys
import pandas as pd

class Recommendation:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_users_one(self,interactions_csr, model, user_id: int):
        try:
            ids, scores = model.recommend(user_id, interactions_csr[user_id], N=5, filter_already_liked_items=False)
            return ids, scores

        except Exception as e:
            raise TrainException(e,sys)

    def get_recommendations_similar_users_all(self,interactions_csr, model):
        try:
            recs_imp = model.recommend_all(user_items=interactions_csr, N=10, filter_already_liked_items=False)
            return recs_imp

        except Exception as e:
            raise TrainException(e,sys)
