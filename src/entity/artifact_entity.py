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
    trained_ineractions_model_file_path: str

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    #train_model_metric_artifact: ClassificationMetricArtifact
    #best_model_metric_artifact: ClassificationMetricArtifact