'''
from src.logger import logging
from src.exception import TrainException
import os,sys
import pandas as pd
from src.components.model_recommendation import Recommendation
class ModelEvaluation:
    def __init__(self):
        try:
            self.model_recommendation = Recommendation()
            self.data_split = DataSplit()
        except Exception as e:
            raise TrainException(e,sys)

    def model_evaluating_similar_users(self):
        try:
           pass
            
        except Exception as e:
            raise TrainException(e,sys)
'''