import os
from random import uniform
from ast import literal_eval
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from src.utils.data_fetch_log_config import logger


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
            options.add_argument("--headless")
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

    def get_news_article_texts(
            self,
            source_path: str = "./data/raw",
            save_path: str = "./data/interim"
        ) -> None:

        self.__configure_webdriver()

        logger.info("loading news data from %s", source_path)
        csv_files = self.__scanDir(
            directory=source_path,
            extension=".csv",
            check_dir=save_path,
            pickup=True
        )
        logger.info("found %s unparsed csv files in %s", len(csv_files), source_path)
        success_count = 0
        for csv_file in csv_files:
            try:
                news_df = pd.read_csv(f"{source_path}/{csv_file}")

                logger.info("getting news article texts for %s", csv_file)
                load_status = True
            except Exception as e:
                load_status = False
                print(f"error loading news data for {csv_file}", e)
                logger.error("error loading news data for %s:\n %s", csv_file, e)

            if load_status:
                try:
                    text_count = 0
                    row_count = 0
                    for index, row in news_df.iterrows():
                        row_texts = str()
                        for link in literal_eval(row["links"]):
                            if len(row_texts) == 0:
                                row_texts += self.__get_link_text(link)
                            else:
                                row_texts += "\n " + self.__get_link_text(link)
                            text_count += 1
                        news_df.loc[index, "link_text"] = row_texts
                        row_count += 1
                            
                            # sleep(uniform(1, 3))

                    news_df.to_csv(f"{save_path}/{csv_file}", index=False)
                    logger.info(
                        "%s updated with %s link texts in %s rows and saved to %s/%s",
                        csv_file,
                        text_count,
                        row_count,
                        save_path,
                        csv_file
                    )
                    success_count += 1
                except Exception as e:
                    print(f"error getting news article texts for {csv_file}", e)
                    logger.error("error getting news article texts for %s:\n %s", csv_file, e)
        logger.info("fetched news article texts for %s date ranges", success_count)
        self.wd.quit()


    def __configure_webdriver(self) -> None:
        try:
            options = webdriver.FirefoxOptions()
            options.page_load_strategy = 'none'
            options.add_argument("--headless")
            self.wd = webdriver.Firefox(options=options)
            self.wd.maximize_window()
            logger.info("webdriver configured")
        except Exception as e:
            print("error configuring webdriver", e)
            logger.error("error configuring webdriver:\n %s", e)


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
        
    def __get_link_text(self, link: str) -> str:
        try:
            # Open a new tab
            self.wd.execute_script("window.open('');")

            # Switch to the new tab
            self.wd.switch_to.window(self.wd.window_handles[-1])

            # Navigate to the link
            self.wd.get(link[:-4])
            WebDriverWait(self.wd, uniform(20,30)).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "news__caption"))
            )
            self.wd.execute_script("window.stop();")

            # check source
            source = self.wd.find_element(
                By.CLASS_NAME,
                "news__caption"
            ).text

            if "youtube" not in source.lower():
                # Extract the content
                link_text = self.wd.find_element(By.CLASS_NAME, 'news__copy').text

                # Close the tab
                self.wd.close()

                # Switch back to the original tab
                self.wd.switch_to.window(self.wd.window_handles[0])

                return link_text
            else:
                # Close the tab
                self.wd.close()
                
                # Switch back to the original tab
                self.wd.switch_to.window(self.wd.window_handles[0])
                print("youtube link, skipping...")
                return ""

        except Exception as e:
            # Close the tab
            self.wd.close()

            # Switch back to the original tab
            self.wd.switch_to.window(self.wd.window_handles[0])
            print("error getting link text", e)
            self.wd.quit()
            self.__configure_webdriver()
            return ""


    @staticmethod
    def __scanDir(directory: str, extension: str, check_dir: str = None, pickup: bool = False) -> list[str]:
        """Check specified directory and return list of files with
        specified extension

        Args:
            extension (str): extension type to be searched for e.g. ".txt"

        Returns:
            list: strings of file names with specified extension
        """    
        files: list = []
        if pickup:
            check_list = os.listdir(check_dir)
            for filename in os.listdir(directory):
                if filename.endswith(extension) and "news" in filename and filename not in check_list:
                    files.append(filename)

        else:
            for filename in os.listdir(directory):
                if filename.endswith(extension) and "news" in filename:
                    files.append(filename)
        files.sort()
        return files
    
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

    @staticmethod
    def __clean_date_time(df: pd.DataFrame) -> pd.DataFrame:
        # Handle leading rows with incorrect timestamps
        first_valid_index = df[df["timestamp"].str.contains("am|pm", na=False)].index.min()
        if first_valid_index is not None:
            first_valid_timestamp = df.loc[first_valid_index, "timestamp"]
            for index in range(first_valid_index):
                df.loc[index, "timestamp"] = first_valid_timestamp

        # Iterate through the rest of the rows to fix missing timestamps
        for index, row in df.iterrows():
            if 'am' not in str(row["timestamp"]) and 'pm' not in str(row["timestamp"]):
                if index > 0:
                    # Replace missing timestamp with the previous row's timestamp
                    df.loc[index, "timestamp"] = df["timestamp"].iloc[index - 1]

            # Combine 'date' and 'timestamp' columns into a single string
            date_string = row["date"].replace('\n', ' ') + " " + str(df.loc[index, "timestamp"])
            df.loc[index, "datetime"] = datetime.strptime(date_string, '%a %b %d %Y %I:%M%p')

        return df


    def combine_news_csv_files(
            self,
            source_path: str = "./data/interim",
            save_path: str = "./data/interim") -> None:
        
        files = self.__scanDir(source_path, ".csv")
        logger.info("found %s csv files", len(files))

        dataframes = []
        skipped_files = []
        for file in files:
            try:
                raw_df = pd.read_csv(f"{source_path}/{file}")
                if len(raw_df) != 0:
                    dataframes.append(self.__clean_date_time(raw_df))
                else:
                    skipped_files.append(file)
                    logger.warning("skipping empty csv file %s", file)
            except Exception as e:
                logger.warning("error loading csv file %s:\n %s", file, e)
                skipped_files.append(file)
                continue
        logger.info("combined %s csv files", len(dataframes))
        logger.warning("skipped %s csv files", len(skipped_files))

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(f"{save_path}/combined_scrapped_data.csv", index=False)
        logger.info("combined news data saved to %s/combined_scraped_data.csv", save_path)


if __name__ == "__main__":
    news_data = NewsData(
        symbols=["EUR", "USD"],
        save_path="./data/raw"
        )

    # ! picking up from date range index 188 based on log file end date
    # ! change this if you want to start from a different date
    # Fetch Price Data
    # news_data.scrape_news(
    #     start_year=2014,
    #     end_year=2024,
    #     pickup_idx=574
    # )

    # Scrape News Data
    # news_data.get_news_article_texts()
    
    # Combine News Data
    news_data.combine_news_csv_files()