from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
        
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
                print("Training pipeline is already running.")

        train_pipeline.run_pipeline()
        print("Training successful !!")
        
