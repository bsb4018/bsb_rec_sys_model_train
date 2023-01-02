from src.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    getdataobj = DataIngestion()
    print(getdataobj.get_interaction_features_from_feature_store())
    #print(getdataobj.get_interaction_features_from_feature_store_online(462769))
    #print(getdataobj.get_course_features_from_feature_store())
    #print(getdataobj.get_course_features_from_feature_store_online(123))