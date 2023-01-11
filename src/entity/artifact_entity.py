from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_interactions_file_path: str
    test_interactions_file_path: str
    interactions_all_data_file_path: str
    courses_all_data_file_path: str

@dataclass
class ModelTrainerArtifact:
    trained_courses_model_file_path: str
    trained_interactions_model_file_path: str
    interactions_matrix_file_path: str

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_hitrate: float
    best_model_path: str
    best_model_report_path: str
    current_interactions_model_path: str
    current_interactions_model_report_file_path: str
    current_courses_model_path:str
    interactions_matrix_file_path: str
    

@dataclass
class ModelPusherArtifact:
    model_file_path:str
    best_interactions_model_file:str
    courses_model_file:str
    saved_best_interactions_model_file:str
    saved_courses_model_file:str
    saved_interactions_matrix_file_path:str

    