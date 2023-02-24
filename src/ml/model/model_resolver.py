import os
from src.constants.training_pipeline import SAVED_MODEL_DIR, PRODUCTION_INTERACTIONS_MODEL_FILE_NAME, PRODUCTION_INTERACTIONS_MODEL_REPORT_FILE_NAME

class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e

    def get_best_model_path(self,) -> str:
        '''Returns the best model path'''
        try:
            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", PRODUCTION_INTERACTIONS_MODEL_FILE_NAME)
            return latest_model_path
        
        except Exception as e:
            raise e

    def is_model_exists(self) -> bool:
        '''Check if a model exists in the list of best models'''
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False

            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False

            return True

        except Exception as e:
            raise e
    
    def get_best_model_report_path(self,) -> str:
        '''Returns the best model evaluation report file'''
        try:
            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}",  PRODUCTION_INTERACTIONS_MODEL_REPORT_FILE_NAME)
            return latest_model_path
        
        except Exception as e:
            raise e
    
    