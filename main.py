from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    #print(getdataobj.get_interaction_features_from_feature_store())
    #print(getdataobj.get_interaction_features_from_feature_store_online(462769))
    #print(getdataobj.get_course_features_from_feature_store())
    #print(getdataobj.get_course_features_from_feature_store_online(123))
    #print(gettrainobj.load_interactions_ingested_features())
    #recs = gettrainobj.get_recommendations("JavaScript: Understanding the Weird Parts")
    #print(recs)

    if train_pipeline.is_pipeline_running:
            print("Training pipeline is already running")
            exit()
    
    train_pipeline.run_pipeline()

    