from src.pipeline.train_pipeline import TrainPipeline
import logging
import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == "__main__":
    main()
