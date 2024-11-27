from datetime import datetime, timedelta
from time import sleep
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
from bs4 import BeautifulSoup as bs
from src.utils.data_log_config import logger


class NewsData:
    def __init__(self, symbols: list[str], save_path: str = None):
        self.symbols = symbols
        self.save_path = save_path


    def scrape_news(self, start_year: int, end_year: int, pickup_idx: int = None) -> None:

        date_ranges = self.__generate_date_ranges(start_year, end_year)

        if pickup_idx is not None:
            date_ranges = date_ranges[pickup_idx:]

        for date_range in date_ranges:
            options = webdriver.FirefoxOptions()
            options.page_load_strategy = 'none'
            # options.add_argument("--headless")
            self.wd = webdriver.Firefox(options=options)
            self.wd.maximize_window()
            news_list = []
            try:
                url = f"https://www.forexfactory.com/calendar?range={date_range}"
                self.__get_page(url)
                day_news = self.__get_day_news()
                for day in day_news:
                    rows, date = self.__get_news_rows(day)
                    for row in rows:
                        data = self.__get_row_data(row)
                        if data["symbol"] in self.symbols:
                            data["date"] = date + " " + date_range[-4:]
                            news_list.append(data)

                logger.info("%s news data fetched for %s", len(news_list), date_range)
                pd.DataFrame(news_list).to_csv(f"{self.save_path}/news_{date_range}.csv", index=False)
                logger.info("news data saved to news_%s.csv", date_range)
                self.wd.quit()
            except Exception as e:
                logger.info("%s news data fetched for %s", len(news_list), date_range)
                pd.DataFrame(news_list).to_csv(f"{self.save_path}/news_{date_range}.csv", index=False)
                logger.info("news data saved to news_%s.csv", date_range)
                print(f"error fetching news data for {date_range}", e)
                logger.error("error fetching news data for %s:\n %s", date_range, e)
                self.wd.quit()


    def __get_page(self, url: str) -> WebElement:
        print("i got to __get_page")
        try:
            logger.info("loading page -> %s", url)
            self.wd.delete_all_cookies()
            self.wd.get(url)

            WebDriverWait(self.wd, 30).until(
                lambda wd: wd.execute_script("return document.readyState") == "complete"
            )

            WebDriverWait(self.wd, 30).until(
                EC.presence_of_all_elements_located((
                    By.CSS_SELECTOR,
                    "td.calendar__cell.calendar__impact span[title]"))
            )
            self.wd.execute_script("window.stop();")

            logger.info("page loaded -> %s", url)
            return self.wd
        except Exception as e:
            print(f"error loading page -> {url}", e)
            logger.error("error loading page -> %s:\n %s", url, e)
            return self.wd

    def __get_day_news(self) -> list[WebElement]:
        print ("i got to __get_day_news")
        return self.wd.find_elements(
            By.TAG_NAME,
            "tbody"
        )

    def __get_news_rows(self, day_news: WebElement) -> tuple[list[WebElement], str]:
        print("i got to __get_news_rows")
        try:
            rows = day_news.find_elements(
                By.TAG_NAME,
                "tr"
            )
            date = day_news.find_element(
                By.CLASS_NAME,
                "date"
            ).text

            return rows, date
        except Exception as e:
            print("error getting news rows", e)
            return [], ""

    def __get_row_data(self, row: WebElement) -> dict:
        print("i got to __get_row_data")
        try:
            symbol = row.find_element(
                By.CLASS_NAME,
                "calendar__currency"
            )
            print("i got past symbol")
            if symbol.text in self.symbols:
                timestamp = row.find_element(
                                By.CLASS_NAME,
                                "calendar__time"
                            ).text
                print("i got past timestamp")
                impact = row.find_element(
                            By.CSS_SELECTOR,
                            "td.calendar__cell.calendar__impact span[title]"
                        ).get_attribute("title")
                print("i got past impact")
                links = self.__get_row_links(row)

                return {
                    "date": "",
                    "timestamp": timestamp,
                    "symbol": symbol.text,
                    "impact": impact,
                    "links": links
                }
            else:
                return {
                    "date": "",
                    "timestamp": "",
                    "symbol": "",
                    "impact": "",
                    "links": []
                }
        except Exception as e:
            print("error getting row data", e)
            return {
                    "date": "",
                    "timestamp": "",
                    "symbol": "",
                    "impact": "",
                    "links": []
                }

    def __get_row_links(self, row: WebElement) -> list[str]:
        print("i got to __get_row_links")
        try:
            # click the detail button
            row.find_element(By.CLASS_NAME, "calendar__detail").click()
            WebDriverWait(self.wd, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "flexposts__storydisplay-info"))
            )
            # self.wd.execute_script("window.stop();")

            # get latest related stories
            last_related_stories = self.wd.find_elements(
                By.CLASS_NAME,
                "relatedstories"
            )[-1]

            # get news flexboxes
            flexboxes = last_related_stories.find_elements(
                            By.CLASS_NAME,
                            "flexposts__storydisplay-info"
                        )

            # get links
            links = []
            for flexbox in flexboxes:
                links.append(
                    flexbox.find_element(
                        By.TAG_NAME,
                        "a"
                    ).get_attribute("href")
                )

            return links
        except Exception as e:
            print("error getting row links", e)
            return []
    
    @staticmethod
    def __generate_date_ranges(start_year: int, end_year: int) -> list[str]:
        date_ranges = []

        # Iterate year by year
        for year in range(start_year, end_year + 1):
            # Start at the beginning of the current year
            current_date = datetime(year, 1, 1)
            # End at the end of the current year
            year_end_date = datetime(year, 12, 31)

            # Generate weekly ranges within the year
            while current_date <= year_end_date:
                # Start of the week
                start_date_str = current_date.strftime("%b%d.%Y").lower()

                # End of the week (7 days later, capped at year_end_date)
                end_of_week = current_date + timedelta(days=6)
                if end_of_week > year_end_date:
                    end_of_week = year_end_date  # Ensure the range doesn't exceed the year's final date
                end_date_str = end_of_week.strftime("%b%d.%Y").lower()

                # Add the range to the list
                date_ranges.append(f"{start_date_str}-{end_date_str}")

                # Move to the next week
                current_date += timedelta(days=7)

        return date_ranges
    
if __name__ == "__main__":
    news_data = NewsData(
        symbols=["EUR", "USD"],
        save_path="./data/raw"
        )
    
    # ! picking up from date range 188 based on log file
    news_data.scrape_news(
        start_year=2014,
        end_year=2024,
        pickup_idx=188
    )