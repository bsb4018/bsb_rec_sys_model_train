from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_interactions_file_path: str
    test_interactions_file_path: str
    interactions_all_data_file_path: str
    courses_all_data_file_path: str
