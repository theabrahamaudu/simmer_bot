Simmer Bot
==============================
Simmer Bot is a machine-learning-powered trading bot that leverages a stacked system of supervised learning models. These models are trained on price data and quantified news data derived from LLM-powered news sentiment analysis.

The overall vision for this project is to create a trading bot capable of ingesting real-time news and price data, analyzing this information, and executing automated trades on any trading platform it is integrated with.

In its current version, the project focuses on defining the architecture for data ingestion, model stack training, and backtesting on unseen data to evaluate model performance.

The model is already slightly profitable on backtest data. However, to achieve optimal results, the selection of currency pairs, data sources, indicator sets, and trading strategies must be refined. This optimization process needs to draw upon seasoned foreign exchange trading experience to ensure the bot's success in dynamic market conditions.

#### Performance Stats for Nerds
Here are the backtesting stats ($300 starting equity) for the current version (Dec 31, 2024):  

    Start                   2024-07-09 05:00:00
    End                     2024-11-15 23:45:00
    Duration                129 days 18:45:00
    Exposure Time [%]       12.7
    Equity Final [$]        309.5973184451607
    Equity Peak [$]         311.0096507378308
    Return [%]              3.199106148386894
    Buy & Hold Return [%]   -2.6074498567335307
    Return (Ann.) [%]       8.808581042303398
    Volatility (Ann.) [%]   2.4929384204544545
    Sharpe Ratio            3.5334130077298993
    Sortino Ratio           8.974576255012797
    Calmar Ratio            9.992983656702695
    Max. Drawdown [%]       -0.8814765784586398
    Avg. Drawdown [%]       -0.10539614291878599
    Max. Drawdown Duration  12 days 18:15:00
    Avg. Drawdown Duration  0 days 17:34:00
    # Trades                29
    Win Rate [%]            58.620689655172406
    Best Trade [%]          0.6003651238405294
    Worst Trade [%]         -0.22458325304276494
    Avg. Trade [%]          0.10890107823988693
    Max. Trade Duration     2 days 14:30:00
    Avg. Trade Duration     0 days 16:14:00
    Profit Factor           2.288900517483482
    Expectancy [%]          0.1093349566311605
    SQN                     1.9431168050607643

## System Architecture
Below is the project architecture. Each shaded area represents a separate module.

![image](https://github.com/theabrahamaudu/simmer_bot/blob/main/docs/Simmer%20Bot%20Architecture%20w%20Logo.png)

## Run Project Locally

Spin up an instannce of Simmer Bot on your local machine by following these steps:
##### N.B: Requires Python 3.10.xx, Windows >= 10, MetaTrader5, Ollama and Firefox browser

- Clone this repository
    ~~~
    git clone https://github.com/theabrahamaudu/simmer_bot.git
    ~~~
- Create a virtual environment
- Install [MetaTrader5](https://download.mql5.com/cdn/web/12018/mt5/hfmarketsglobal5setup.exe) 
- Install [Ollama](https://ollama.com/download/OllamaSetup.exe)
- Set MT5 historical data limit:  
    Go to `Tools` > `Options`. In the next window, select the `Charts` tab at the top, then select `Unlimited` under `Max bars in chart`. Click the `OK` button to save your settings.

- Download the LLM for offline inference generation:
    ~~~
    ollama pull mistral-nemo
    ~~~
- Within the simmer bot root diectory, create a `.env` file and add your trading account login details (Demo account recommended) in this format:
    ~~~
    LOGIN="88888888"
    PASSWORD="YourP@ssw0rd"
    SERVER="YourBrokerServer"
    ~~~
- Within the simmer bot root directory, open `./src/main.py` and set all the stage flags to `True`
- Configure Pipeline:
    - In STAGE 05, before running for the first time,  
        - set `data_path` parameter of `TrainInferencePreprocess` to `"./data/interim/merged_scrapped_n_price.csv"`

        - set `with_llm_sentiment` to `True`, and

        - set `mock` to `False` if you want to use actual LLM sentiment scores (setting mock to `True` uses neutral sentiment score of 0.5 for all news articles).

    - In STAGE 06, the feature selection pipeline defaults to top 20 features. You can change this by setting the`top_k` parameter to your desired number of top features e.g. 
        ~~~
        select_features = SelectFeatures(
            top_k=30
        )
        ~~~

    - In STAGE 08, you can modify the trading account parameters by adding them to the `RunBacktest` initialization e.g.
        ~~~
            # Default
            backtest_pipeline = RunBacktest(
                predictions=PREDICTIONS,
                strategy=MyStrategy
            )

            # Modified
            backtest_pipeline = RunBacktest(
                predictions=PREDICTIONS,
                strategy=MyStrategy,
                cash=800,           # in dollars
                margin= 1 / 3,      # leverage ratio
                commission=0.0004,  # broker commissions
            )
        ~~~
    - Navigate to `./config/config.yaml`. There, you can:
        - Change the currency ticker
        - Change the timeframe (Only supports minute timeframes <= 30)
            ~~~
            TIMEFRAME_M1                        = 1
            TIMEFRAME_M2                        = 2
            TIMEFRAME_M3                        = 3
            TIMEFRAME_M4                        = 4
            TIMEFRAME_M5                        = 5
            TIMEFRAME_M6                        = 6
            TIMEFRAME_M10                       = 10
            TIMEFRAME_M12                       = 12
            TIMEFRAME_M15                       = 15
            TIMEFRAME_M20                       = 20
            TIMEFRAME_M30                       = 30
            ~~~
        - Edit the start and end date for price and news data collection
        - Choose to use mock or real LLM sentiments during backtesting

- Start the pipeline:
    ~~~
    python ./src/main.py
    ~~~

## Description
Simmer Bot is designed to be a robust algorithmic trading bot powered by a machine-learning-based trade-triggering system. It will be integrated into a trading platform and deployed on the cloud, enabling it to automatically ingest price and news data, process the information, and make predictions about future price movements. These predictions will then pass through a set of threshold filters to decide whether to buy, sell, or wait for more definitive market conditions.

Currently, the model stack examines market information from the previous six timesteps to predict the highest high price over the next five timesteps. During the preprocessing phase, hundreds of features are generated, and the feature selection pipeline identifies the top 20 features that most influence the prediction of the highest high price. Unfortunately, in the current iteration, news sentiment is filtered out due to its relatively low importance (ranking around 200 out of ~945 features). This highlights the need for either a higher-quality source of news data or improvements to the news scraping mechanism to capture more detailed information.

Price data is fetched from HotForex using a demo account, while news data is collected from ForexFactory via a custom web scraper. At present, the scraper gathers information from the news summary page, but it may need to be enhanced to scrape full article pages for deeper insights.

The trading rules currently use the predicted highest high price as the take-profit price when placing trades, provided it is significantly higher than the current market price.

As outlined in the [Project Plan](https://github.com/theabrahamaudu/simmer_bot/blob/main/project_plan.md), the next steps involve achieving a more performant model stack. Once this is accomplished, the bot will be integrated into a trading platform to execute trades, starting with a demo account and eventually transitioning to live trading. Subsequently, the entire architecture will be deployed to the cloud to ensure uninterrupted operation. This deployment will necessitate further efforts in monitoring and retraining the model to adapt to shifting market conditions over time.

## Basic Model Training & Evaluation Workflow
- Fetch price data
- Fetch news data
- Clean news data
- Merge news and price data
- Preprocess data:
    - LLM sentiment on news data
    - TA indicators on price data
    - Data manipulation for machine learning 
- Feature selection
- Model stack training
- Model & trading strategy backtesting

## Dependencies
#### OS Level
- Windows >=10 (Developed and tested on Windows 11)
- MetaTrader5
- Ollama
- Firefox (Headless mode)

#### Python (3.10.11)
    click
    Sphinx
    coverage
    flake8
    python-dotenv>=0.5.1
    ipykernel
    ipywidgets
    MetaTrader5==5.0.4424
    ta_lib-0.5.1-cp310-cp310-win_amd64.whl
    tqdm==4.67.1
    pandas==2.2.3
    numpy==1.26.4
    matplotlib==3.10.0
    seaborn==0.13.2
    langchain-community==0.3.9
    selenium==4.26.1
    beautifulsoup4==4.12.3
    pyyaml==6.0.2
    scikit-learn==1.5.2
    "tensorflow<2.11"
    xgboost==2.1.3
    lightgbm==4.5.0
    Backtesting==0.3.3

## Installing
Refer to the [Run Project Locally](#run-project-locally) section.


## Help
Feel free to reach out to me or create a new issue if you encounter any problems setting up or running the Simmer Bot pipeline.

## Possible Improvements/Ideas

- [ ] Unit tests
- [ ] Architecture and hyperparameter tuning for the different models used
- [ ] Higher quality news data
- [ ] Fine-tuned set of TA indicators
- [ ] Fine-tuned trading strategy with model
- [ ] Integration with trading platform
- [ ] Cloud deployment - with monitoring and scheduled model retraining 

## Authors

Contributors names and contact info

*Abraham Audu*

* GitHub - [@the_abrahamaudu](https://github.com/theabrahamaudu)
* X (formerly Twitter) - [@the_abrahamaudu](https://x.com/the_abrahamaudu)
* LinkedIn - [@theabrahamaudu](https://www.linkedin.com/in/theabrahamaudu/)
* Instagram - [@the_abrahamaudu](https://www.instagram.com/the_abrahamaudu/)
* YouTube - [@DataCodePy](https://www.youtube.com/@DataCodePy)

## Version History

* See [commit change](https://github.com/theabrahamaudu/simmer_bot/commits/main/)
* See [release history](https://github.com/theabrahamaudu/simmer_bot/releases)

## Acknowledgments

* This [paper](https://ideas.repec.org/p/arx/papers/2107.14092.html) by Yunze Li & Yanan Xie & Chen Yu & Fangxing Yu & Bo Jiang & Matloob Khushi, 2021 guided the development of the feature selection process, stacked model architecture and initial set of TA indicators used.
* [ChatGPT](chat.openai.com) assisted with drafting the elaborate [Project Plan](https://github.com/theabrahamaudu/simmer_bot/blob/main/project_plan.md).
