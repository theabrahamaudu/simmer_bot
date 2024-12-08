import os
import time
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
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
    
if __name__ == "__main__":
    llm_sentiment_analyzer = LLMSentimentAnalysis()
    llm_sentiment_analyzer.parse_dataframe()