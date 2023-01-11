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

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Into the initiate_model_pusher function of ModelPusher class")
            trained_interactions_model_path = self.model_eval_artifact.current_interactions_model_path
            courses_model_path = self.model_eval_artifact.current_courses_model_path
            current_interactions_model_report_path = self.model_eval_artifact.current_interactions_model_report_file_path

            best_interactions_model = self.model_eval_artifact.best_model_path
            best_interactions_model_report = self.model_eval_artifact.best_model_report_path

            #Pushing the trained interaction model and the courses model in the model pusher directory
            model_file_path = self.model_pusher_config.model_pusher_dir
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.model_pusher_config.best_interactions_model_file),exist_ok=True)
            shutil.copy(src=trained_interactions_model_path, dst=self.model_pusher_config.best_interactions_model_file)
            os.makedirs(os.path.dirname(self.model_pusher_config.best_courses_model_file),exist_ok=True)
            shutil.copy(src=courses_model_path, dst=self.model_pusher_config.best_courses_model_file)
            os.makedirs(os.path.dirname(self.model_pusher_config.best_interactions_model_report_file),exist_ok=True)
            shutil.copy(src=current_interactions_model_report_path, dst=self.model_pusher_config.best_interactions_model_report_file)

            #Pushing the trained interaction model and the courses model in the saved for production directory
            production_model_file_path = self.model_pusher_config.saved_production_model_file_path
            os.makedirs(os.path.dirname(production_model_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.model_pusher_config.saved_production_interactions_model_file),exist_ok=True)
            shutil.copy(src=best_interactions_model, dst=self.model_pusher_config.saved_production_interactions_model_file)
            os.makedirs(os.path.dirname(self.model_pusher_config.saved_production_courses_model_file),exist_ok=True)
            shutil.copy(src=courses_model_path, dst=self.model_pusher_config.saved_production_courses_model_file)
            os.makedirs(os.path.dirname(self.model_pusher_config.saved_production_interactions_model_report_file),exist_ok=True)
            shutil.copy(src=best_interactions_model_report, dst=self.model_pusher_config.saved_production_interactions_model_report_file)

            #Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(model_file_path=model_file_path, \
                best_interactions_model_file=self.model_pusher_config.best_interactions_model_file, \
                    courses_model_file=self.model_pusher_config.best_courses_model_file,\
                        saved_best_interactions_model_file=self.model_pusher_config.saved_production_interactions_model_file,\
                        saved_courses_model_file=self.model_pusher_config.saved_production_courses_model_file
                    )
            return model_pusher_artifact

        except Exception as e:
            raise TrainException(e,sys)
