from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer

if __name__ == "__main__":
    getdataobj = DataIngestion()
    gettrainobj = ModelTrainer()
    #print(getdataobj.get_interaction_features_from_feature_store())
    #print(getdataobj.get_interaction_features_from_feature_store_online(462769))
    #print(getdataobj.get_course_features_from_feature_store())
    #print(getdataobj.get_course_features_from_feature_store_online(123))

    #print(gettrainobj.load_interactions_ingested_features())

    recs = gettrainobj.get_recommendations("JavaScript: Understanding the Weird Parts")
    print(recs)