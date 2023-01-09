from src.exception import TrainException
from src.logger import logging
from src.entity.artifact_entity import ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from src.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
import os,sys
import shutil
from src.utils.main_utils import save_object,load_object,write_yaml_file

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                       model_eval_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except Exception as e:
            raise TrainException(e,sys)

    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:
            logging.info("Into the initiate_model_pusher function of ModelPusher class")
            trained_model_path = self.model_eval_artifact.current_trained_model_path
            trained_model_report_path = self.model_eval_artifact.current_model_report_file_path
            
            #Pushing the trained model in the model storage space
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            #Pushing the trained model in a the saved path for production
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            #Pushing the trained model report in a the saved path for production
            saved_model_report_path = self.model_pusher_config.saved_model_report_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_report_path, dst=saved_model_report_path)

            #Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, saved_model_report_path=saved_model_report_path, model_file_path=model_file_path)
            return model_pusher_artifact

        except Exception as e:
            raise TrainException(e,sys)
