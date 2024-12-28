from .preprocess_pipeline import TrainPreprocessPipeline

if __name__ == "__main__":
    pipeline = TrainPreprocessPipeline()
    pipeline.run(with_llm_sentiment=True)