import os
import time
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import talib as ta
from langchain_core.runnables import RunnableSerializable
from langchain_community.llms import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from src.utils.data_preprocess_log_config import logger

class LLMSentimentAnalysis:
    def __init__(
            self,
            model_name: str= "mistral-nemo",
            source_file_path: str = "./data/interim/merged_scrapped_n_price.csv",
            save_path: str = "./data/interim"
            ):
        logger.info("initializing news sentiment analyzer")
        self.model_name = model_name
        self.source_file_path = source_file_path
        self.save_path = save_path
        self.__prompt_template = """
            You are an expert financial analyst specialized in currency markets, particularly in evaluating
            the impact of news articles on the EURUSD currency pair. Your task is to read the following news
            article and analyze its sentiment to determine its bias toward either the Euro (EUR) or
            the US Dollar (USD).

            Provide a single floating-point value between 0 and 1 that represents this bias:
            - 0.0000: Extremely bullish for the Euro (EUR)
            - 1.0000: Extremely bullish for the US Dollar (USD)
            - 0.5000: Neutral, with no bias toward either currency

            **Instructions:**
            1. Assess the content of the article for references to economic indicators, policies, or events that could impact the EURUSD currency pair.
            2. Weigh any implications of the news against typical market reactions to similar news for the EUR and USD.
            3. Avoid explanations or reasoning in your response; provide only the float value to four decimal places.

            Example Input:\n
            Nonfarm business sector labor productivity increased 3.4 percent in the first quarter of 2019,
            the U.S. Bureau of Labor Statistics reported today, as output increased 3.9 percent and hours
            worked increased 0.5 percent. (All quarterly percent changes in this release are seasonally
            adjusted annual rates.) From the first quarter of 2018 to the first quarter of 2019, productivity
            increased 2.4 percent, reflecting a 3.9- percent increase in output and a 1.5-percent increase in
            hours worked. (See table A1.) The four-quarter increase in productivity is the largest since a
            2.7-percent gain in the third quarter of 2010. Labor ... (full story)


            Example Output:\n
            0.9000

            \n\n

            Article:\n {article}?\n

            Answer:
        """

    def parse_dataframe(self, pickup: bool=False, pickup_index: int=None) -> None:
        logger.info("loading data from %s", self.source_file_path)
        try:
            df = pd.read_csv(self.source_file_path)
            logger.info("loaded data from %s", self.source_file_path)
        except Exception as e:
            logger.error("error loading data from %s:\n %s", self.source_file_path, e)
        model_calls = 0

        if pickup:
            df = df[pickup_index:]

        try:
            start_time = time.time()
            for index, _ in tqdm(
                df.iterrows(),
                total=df.shape[0],
                desc="Parsing news articles...",
                unit="article(s)"
                ):
                
                if index != 0 \
                    and df.loc[index, "link_text"] == df.loc[index-1, "link_text"]:
                    df.loc[index, "sentiment_score"] = df.loc[index-1, "sentiment_score"]
                else:
                    df.loc[index, "sentiment_score"] = \
                        self.news_sentiment(str(df.loc[index, "link_text"]))
                    model_calls += 1
                    if model_calls % 20 == 0:
                        logger.info("model calls: %d", model_calls)
                        logger.info("current index: %d", index)
                        df.to_csv(self.save_path + "/parsed_scraped_data_temp.csv", index=False)
            elapsed_time = time.time() - start_time
            logger.info(
                "parsed %d news articles in %.2f seconds (%.2f articles per second)",
                df.shape[0],
                elapsed_time,
                df.shape[0] / elapsed_time
            )
            logger.info("model calls: %d", model_calls)
        except Exception as e:
            logger.error("error parsing news articles:\n %s", e)
            logger.error("failed at index %d", index)

        try:
            os.makedirs(self.save_path, exist_ok=True)
            logger.info("saving parsed data to %s", self.save_path)
            df.to_csv(self.save_path + "/parsed_scraped_data.csv", index=False)
            logger.info(
                "parsed data saved to %s",
                self.save_path+"/parsed_scraped_data.csv"
            )
        except Exception as e:
            logger.error(
                "error saving parsed data to %s:\n %s",
                self.save_path+"/parsed_scraped_data.csv",
                e
            )
        

    def __get_response_chain(self) -> RunnableSerializable:
        model = ollama.Ollama(
            model=self.model_name,
            temperature=0.0,
            verbose=True
        )

        prompt = PromptTemplate(
            template=self.__prompt_template,
            input_variables=["article"]
        )

        chain = prompt | model | StrOutputParser()
        return chain
    
    def news_sentiment(self, article: str) -> np.float32:
        chain = self.__get_response_chain()
        response = chain.invoke(
            {
                "article": self.__clean_text(article)
            },
            return_only_outputs=True
        )
        try:
            response = np.float32(response)
            if response > 1.0000:
                response = np.float32(1.0000)
            elif response < 0.0000:
                response = 0
            return response
        except Exception as e:
            logger.warning(
                "error parsing news article >> %s <<:\n %s || sentiment set to 0.5",
                article,
                e
            )
            return np.float32(0.5)


    @staticmethod
    def __clean_text(text: str) -> str:
        return re.sub(r'[^\x20-\x7E]+', ' ', text)[:7000]


class TAIndicators:
    def __init__(
            self,
            source_file_path: str = "./data/interim/parsed_scraped_data_clipped.csv",
            save_path: str = "./data/interim"
        ) -> None:
        self.__source_file_path = source_file_path
        self.__save_path = save_path

    def __load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.__source_file_path)
    
    def add_indicators(self) -> pd.DataFrame:
        data = self.__load_data()

        # ADX - Average Directional Movement Index
        data['adx_14'] = ta.ADX(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # ADXR - Average Directional Movement Index Rating
        data['adxr_14'] = ta.ADXR(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # APO - Absolute Price Oscillator
        data['apo_12_26'] = ta.APO(
            data['close'], fastperiod=12, slowperiod=26, matype=0
        )

        data['apo_5_15'] = ta.APO(
            data['close'], fastperiod=5, slowperiod=15, matype=0
        )

        data['apo_8_21'] = ta.APO(
            data['close'], fastperiod=8, slowperiod=21, matype=0
        )

        # AROON - Aroon
        data['aroondown_14'], data['aroonup_14'] = ta.AROON(
            data['high'], data['low'], timeperiod=14
        )

        # AROONOSC - Aroon Oscillator
        data['aroon_osc_14'] = ta.AROONOSC(
            data['high'], data['low'], timeperiod=14
        )

        # BOP - Balance Of Power
        data['bop'] = ta.BOP(
            data['open'], data['high'], data['low'], data['close']
        )

        # CCI - Commodity Channel Index
        data['cci_14'] = ta.CCI(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # CMO - Chande Momentum Oscillator
        data['cmo_14'] = ta.CMO(data['close'], timeperiod=14)

        # DX - Directional Movement Index
        data['dx_14'] = ta.DX(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # MACD - Moving Average Convergence/Divergence
        data['macd_12_26_9'], data['macdsignal_12_26_9'], data['macdhist_12_26_9'] = ta.MACD(
            data['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        data['macd_5_13_6'], data['macdsignal_5_13_6'], data['macdhist_5_13_6'] = ta.MACD(
            data['close'],
            fastperiod=5,
            slowperiod=13,
            signalperiod=6
        )

        # MACDEXT - MACD with controllable MA type
        data['macd_ext_12_26_9'], data['macdsignal_ext_12_26_9'], data['macdhist_ext_12_26_9'] = ta.MACDEXT(
            data['close'],
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0
        )
        
        data['macd_ext_5_13_6'], data['macdsignal_ext_5_13_6'], data['macdhist_ext_5_13_6'] = ta.MACDEXT(
            data['close'],
            fastperiod=5,
            fastmatype=0,
            slowperiod=13,
            slowmatype=0,
            signalperiod=6,
            signalmatype=0
        )
        
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        data['macd_fix'], data['macdsignal_fix'], data['macdhist_fix'] = ta.MACDFIX(
            data['close'], signalperiod=9
        )

        # MINUS_DI - Minus Directional Indicator
        data['minus_di_14'] = ta.MINUS_DI(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # MINUS_DM - Minus Directional Movement
        data['minus_dm_14'] = ta.MINUS_DM(
            data['high'], data['low'], timeperiod=14
        )

        # MOM - Momentum
        data['mom_10'] = ta.MOM(data['close'], timeperiod=10)
        data['mom_9'] = ta.MOM(data['close'], timeperiod=9)
        data['mom_14'] = ta.MOM(data['close'], timeperiod=14)

        # PLUS_DI - Plus Directional Indicator
        data['plus_di_14'] = ta.PLUS_DI(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        # PLUS_DM - Plus Directional Movement
        data['plus_dm_14'] = ta.PLUS_DM(
            data['high'], data['low'], timeperiod=14
        )

        # PPO - Percentage Price Oscillator
        data['ppo_12_26'] = ta.PPO(
            data['close'], fastperiod=12, slowperiod=26, matype=0
        )

        data['ppo_5_13'] = ta.PPO(
            data['close'], fastperiod=5, slowperiod=13, matype=0
        )

        # ROC - Rate of change : ((price/prevPrice)-1)*100
        data['roc_10'] = ta.ROC(data['close'], timeperiod=10)
        data['roc_5'] = ta.ROC(data['close'], timeperiod=5)
        data['roc_14'] = ta.ROC(data['close'], timeperiod=14)

        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        data['rocp_10'] = ta.ROCP(data['close'], timeperiod=10)
        data['rocp_5'] = ta.ROCP(data['close'], timeperiod=5)
        data['rocp_14'] = ta.ROCP(data['close'], timeperiod=14)

        # ROCR - Rate of change ratio: (price/prevPrice)
        data['rocr_10'] = ta.ROCR(data['close'], timeperiod=10)
        data['rocr_5'] = ta.ROCR(data['close'], timeperiod=5)
        data['rocr_14'] = ta.ROCR(data['close'], timeperiod=14)

        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        data['rocr100_10'] = ta.ROCR100(data['close'], timeperiod=10)
        data['rocr100_5'] = ta.ROCR100(data['close'], timeperiod=5)
        data['rocr100_14'] = ta.ROCR100(data['close'], timeperiod=14)

        # RSI - Relative Strength Index
        data['rsi_14'] = ta.RSI(data['close'], timeperiod=14)

        # STOCH - Stochastic
        data['slowk_3'], data['slowd_3'] = ta.STOCH(
            data['high'], data['low'], data['close'],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )

        # STOCHF - Stochastic Fast
        data['fastk_5'], data['fastd_3'] = ta.STOCHF(
            data['high'], data['low'], data['close'],
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0
        )

        # STOCHRSI - Stochastic Relative Strength Index
        data['fastk_rsi_14_5'], data['fastd_rsi_14_3'] = ta.STOCHRSI(
            data['close'],
            timeperiod=14,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0)

        data['fastk_rsi_5_5'], data['fastd_rsi_5_3'] = ta.STOCHRSI(
            data['close'],
            timeperiod=5,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0)

        data['fastk_rsi_10_5'], data['fastd_rsi_10_3'] = ta.STOCHRSI(
            data['close'],
            timeperiod=10,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0)

        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        data['trix_30'] = ta.TRIX(data['close'], timeperiod=30)
        data['trix_5'] = ta.TRIX(data['close'], timeperiod=5)
        data['trix_9'] = ta.TRIX(data['close'], timeperiod=9)
        data['trix_14'] = ta.TRIX(data['close'], timeperiod=14)

        # EMA - Exponential Moving Average
        data['ema_9'] = ta.EMA(data["close"], timeperiod=9)

        # ULTOSC - Ultimate Oscillator
        data['ult_osc_7_14_28'] = ta.ULTOSC(
            data['high'], data['low'], data['close'],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )

        # WILLR - Williams' %R
        data['willr_14'] = ta.WILLR(
            data['high'], data['low'], data['close'], timeperiod=14
        )

        data['willr_7'] = ta.WILLR(
            data['high'], data['low'], data['close'], timeperiod=7
        )

        data['willr_21'] = ta.WILLR(
            data['high'], data['low'], data['close'], timeperiod=21
        )

    
if __name__ == "__main__":
    llm_sentiment_analyzer = LLMSentimentAnalysis()
    llm_sentiment_analyzer.parse_dataframe()