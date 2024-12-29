"""
Main pipeline to run the entire project


"""
import yaml
from time import perf_counter, strftime, gmtime
from src.data.price_data import PriceData
from src.data.news_data import NewsData
from src.data.make_dataset import CleanData, MergeData
from src.features.preprocess_pipeline import TrainPreprocessPipeline
from src.models.train_model import SelectFeatures, TrainStackedModel
from src.backtest.backtest_model import RunBacktest, MyStrategy, PREDICTIONS
from src.utils.main_pipeline_log_config import logger


if __name__ == "__main__":
    config = yaml.safe_load(open("config/config.yaml"))
    STAGE_01 = ["Fetch Price Data", False]
    STAGE_02 = ["Fetch News Data", False]
    STAGE_03 = ["Clean News Data", False]
    STAGE_04 = ["Merge News & Price Data", False]
    STAGE_05 = ["Train Preprocess Pipeline", False]
    STAGE_06 = ["Select Features", True]
    STAGE_07 = ["Train Stacked Model", True]
    STAGE_08 = ["Run Backtest", True]

# -----------------------------------------------------------------------------
#   START
# -----------------------------------------------------------------------------
    print(f"Main pipeline started, check logs for details >> ./logs/main_pipeline.log")
    logger.info("Pipeline started")
    ACTIVE_STAGES = [] 
    for stage, flag in [
        STAGE_01,
        STAGE_02,
        STAGE_03,
        STAGE_04,
        STAGE_05,
        STAGE_06,
        STAGE_07,
        STAGE_08]:
        if flag:
            ACTIVE_STAGES.append(stage)
    logger.info("Running stages: %s", ACTIVE_STAGES)

# -----------------------------------------------------------------------------
#   STAGE 01: FETCH PRICE DATA
# -----------------------------------------------------------------------------

    start_time = perf_counter()
    if STAGE_01[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_01[0])
        try:
            price_data = PriceData(str(config["symbol"]))
            fetched_data = price_data.fetch(
                list(config["start"]),
                list(config["end"]),
                int(config["timeframe"])
            )
            price_data.save(fetched_data, "./data/raw/price_data.csv")
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_01[0])
        except Exception as e:
            logger.error("Error fetching price data: %s", e)
    
    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_01[0])

# -----------------------------------------------------------------------------
#   STAGE 02: FETCH NEWS DATA
# -----------------------------------------------------------------------------

    if STAGE_02[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_02[0])
        try:
            news_data = NewsData(
                symbols=[
                    str(config["symbol"])[:3],
                    str(config["symbol"])[3:]
                ],
                save_path="./data/raw"
            )
            news_data.scrape_news(
                list(config["start"])[0],
                list(config["end"])[0],
            )
            news_data.get_news_article_texts()
            news_data.combine_news_csv_files()  
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_02[0])
        except Exception as e:
            logger.error("Error fetching news data: %s", e)
    
    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_02[0])

# -----------------------------------------------------------------------------
#   STAGE 03: CLEAN NEWS DATA
# -----------------------------------------------------------------------------

    if STAGE_03[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_03[0])
        try:
            news_data_cleaner = CleanData(
                source_file_path="./data/interim/combined_scrapped_data.csv",
                save_path="./data/interim"
            )
            news_data_cleaner.clean()
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_03[0])
        except Exception as e:
            logger.error("Error cleaning news data: %s", e)
    
    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_03[0])

# -----------------------------------------------------------------------------
#   STAGE 04: MERGE NEWS & PRICE DATA
# -----------------------------------------------------------------------------

    if STAGE_04[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_04[0])
        try:
            merge_data = MergeData(
                price_source_path="./data/raw/price_data.csv",
                news_source_path="./data/interim/cleaned_scraped_data.csv",
                save_path="./data/interim"
            )
            merge_data.merge()
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_04[0])
        except Exception as e:
            logger.error("Error merging data: %s", e)
    
    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_04[0])

# -----------------------------------------------------------------------------
#   STAGE 05: TRAIN PREPROCESS PIPELINE
# -----------------------------------------------------------------------------

    if STAGE_05[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_05[0])
        try:
            train_preprocess_pipeline = TrainPreprocessPipeline(
                data_path="./data/interim/parsed_scraped_data_clipped.csv"
            )
            _, _, _ = train_preprocess_pipeline.run()
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_05[0])
        except Exception as e:
            logger.error("Error running preprocess pipeline: %s", e)

    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_05[0])

# -----------------------------------------------------------------------------
#   STAGE 06: SELECT FEATURES
# -----------------------------------------------------------------------------
    FEATURE_LIST_EXISTS = False
    if STAGE_06[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_06[0])
        try:
            select_features = SelectFeatures()
            top_features = select_features.run()
            FEATURE_LIST_EXISTS = True
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_06[0])
        except Exception as e:
            logger.exception("Error selecting features: %s", e)

    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_06[0])

# -----------------------------------------------------------------------------
#   STAGE 07: TRAIN STACKED MODEL
# -----------------------------------------------------------------------------

    if STAGE_07[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_07[0])
        try:
            if FEATURE_LIST_EXISTS:
                train_stacked_model = TrainStackedModel(
                top_k_features=top_features
                )
            else:
                train_stacked_model = TrainStackedModel()
                train_stacked_model.train_stack()

            train_stacked_model.train_stack()
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_07[0])
        except Exception as e:
            logger.error("Error training stacked model: %s", e)

    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_07[0])

# -----------------------------------------------------------------------------
#   STAGE 08: RUN BACKTEST
# -----------------------------------------------------------------------------

    if STAGE_08[1]:
        logger.info(">>>>>>>>>> starting stage: %s <<<<<<<<<<<", STAGE_08[0])
        try:
            backtest_pipeline = RunBacktest(
                predictions=PREDICTIONS,
                strategy=MyStrategy
            )
            backtest_pipeline.run()
            logger.info(">>>>>>>>>> completed stage: %s <<<<<<<<<<<", STAGE_08[0])
        except Exception as e:
            logger.error("Error running backtest: %s", e)

    else:
        logger.info(">>>>>>>>>> skipped stage: %s <<<<<<<<<<<", STAGE_08[0])

# -----------------------------------------------------------------------------
#   END
# -----------------------------------------------------------------------------

    total_run_time = perf_counter() - start_time
    formatted_time = strftime("%H:%M:%S", gmtime(total_run_time))
    logger.info("Pipeline completed in %s", formatted_time)

