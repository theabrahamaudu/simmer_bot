from src.features.preprocess_pipeline import TrainPreprocessPipeline

if __name__ == "__main__":
    pipeline = TrainPreprocessPipeline(

    )
    _, _, _ = pipeline.run(with_llm_sentiment=True)