from datetime import datetime
from collections import namedtuple

TrainingPipelineConfig = namedtuple("PipelineConfig",  ["pipeline_name", "artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig",  ["data_ingestion_dir",
                                                         "all_interactions_file_path",
                                                         "all_users_file_path",
                                                         "interactions_train_file_path",
                                                         "interactions_test_file_path",
                                                         "interactions_split_percentage",
                                                        ])     

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["model_trainer_dir", "trained_interactions_model_file_path", "interactions_matrix_shape_file_path","model_users_map_file_path"
                                                       ])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig",
                                   ["model_evaluation_dir", "report_file_path", "change_threshold"])


ModelPusherConfig = namedtuple("ModelPusherConfig", ["model_pusher_dir", 
                                                     "best_interactions_model_file",\
                                                        "best_interactions_model_report_file",\
                                                            "best_interactions_model_matrix_shape_file",\
                                                              "best_users_map_file",\
                                                                "saved_production_interactions_model_file",\
                                                                    "saved_production_interactions_model_report_file",\
                                                                        "saved_production_interactions_matrix_shape_file",
                                                                        "saved_model_users_map_file_path"
                                                                        ])
