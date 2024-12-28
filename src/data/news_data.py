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
    """
    Class to scrape news data from forex factory
    """
    def __init__(self, symbols: list[str], save_path: str = None):
        self.symbols = symbols
        self.save_path = save_path


    def scrape_news(self, start_year: int, end_year: int, pickup_idx: int = None) -> None:
        """
        Scrapes news data from the Forex Factory website for a specified date range.

        Args:
            start_year (int): The start year of the date range to scrape.
            end_year (int): The end year of the date range to scrape.
            pickup_idx (int, optional): The starting index of the date range. If provided, 
                                        skips earlier ranges.

        Generates date ranges, fetches news data for each range, and saves the data to CSV files.

        Raises:
            Exception: Logs and handles errors encountered during scraping.
        """

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
                logger.error("error fetching news data for %s:\n %s", date_range, e)
                self.wd.quit()

    def get_news_article_texts(
            self,
            source_path: str = "./data/raw",
            save_path: str = "./data/interim"
        ) -> None:
        """
        Extracts and saves the text of news articles from links in raw news data CSV files.

        Args:
            source_path (str): Path to the directory containing raw CSV files. Default is "./data/raw".
            save_path (str): Path to the directory where updated CSV files with article texts 
                            will be saved. Default is "./data/interim".

        Processes each CSV file to fetch article texts from the URLs provided in the "links" column 
        and appends the text to a new "link_text" column in the same file.

        Logs:
            - Number of unparsed CSV files found.
            - Processing status of each file.
            - Number of rows and texts updated.

        Raises:
            Exception: Logs and handles errors during file loading, text extraction, or file saving.
        """
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
                    logger.error("error getting news article texts for %s:\n %s", csv_file, e)
        logger.info("fetched news article texts for %s date ranges", success_count)
        self.wd.quit()


    def __configure_webdriver(self) -> None:
        """
        Configures the Selenium WebDriver for headless Firefox browsing.

        Initializes a headless Firefox WebDriver with a non-blocking page load strategy 
        and maximizes the browser window.

        Logs:
            - Success message upon successful configuration.
            - Error message if an exception occurs during configuration.

        Raises:
            Exception: Logs and handles errors that occur during WebDriver setup.
        """
        try:
            options = webdriver.FirefoxOptions()
            options.page_load_strategy = 'none'
            options.add_argument("--headless")
            self.wd = webdriver.Firefox(options=options)
            self.wd.maximize_window()
            logger.info("webdriver configured")
        except Exception as e:
            logger.error("error configuring webdriver:\n %s", e)


    def __get_page(self, url: str) -> WebElement:
        """
        Loads a webpage in the Selenium WebDriver and waits for it to fully load.

        Args:
            url (str): The URL of the webpage to load.

        Returns:
            WebElement: The WebDriver instance after loading the page.

        Behavior:
            - Clears all cookies and navigates to the specified URL.
            - Waits for the page's `document.readyState` to be "complete".
            - Ensures specific elements are present on the page before proceeding.
            - Stops further page loading after the initial content is ready.

        Logs:
            - Success message when the page is successfully loaded.
            - Error message if an exception occurs during page loading.

        Raises:
            Exception: Logs and handles errors during page loading.
        """
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
            logger.error("error loading page -> %s:\n %s", url, e)
            return self.wd

    def __get_day_news(self) -> list[WebElement]:
        """
        Retrieves daily news elements from the loaded webpage.

        Returns:
            list[WebElement]: A list of WebElements representing the daily news sections 
                            (found within `tbody` tags).

        Logs:
            - Debug message indicating method execution.

        Raises:
            Exception: Any WebDriver-related issues will propagate to the caller.
        """

        return self.wd.find_elements(
            By.TAG_NAME,
            "tbody"
        )

    def __get_news_rows(self, day_news: WebElement) -> tuple[list[WebElement], str]:
        """
        Extracts rows of news data and the corresponding date from a daily news section.

        Args:
            day_news (WebElement): The WebElement representing the daily news section.

        Returns:
            tuple[list[WebElement], str]: 
                - A list of WebElements representing the rows of news data.
                - A string representing the date of the news.

        Logs:
            - Error message if an exception occurs during extraction.

        Raises:
            Exception: Logs and handles errors during row and date extraction.
        """
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
            logger.error("error getting news rows:\n %s", e)
            return [], ""

    def __get_row_data(self, row: WebElement) -> dict:
        """
        Extracts data from a single row in the news section.

        Args:
            row (WebElement): The WebElement representing a row of news data.

        Returns:
            dict: A dictionary containing:
                - "date" (str): Placeholder for the date (populated externally).
                - "timestamp" (str): The time associated with the news.
                - "symbol" (str): The currency symbol related to the news.
                - "impact" (str): The economic impact level (retrieved from the tooltip).
                - "links" (list): A list of URLs extracted from the row.

            If the symbol is not relevant, returns an empty dictionary structure.

        Logs:
            - Debug messages for successful processing steps.
            - Error message if an exception occurs during data extraction.

        Raises:
            Exception: Logs and handles errors during row data extraction.
        """
        try:
            symbol = row.find_element(
                By.CLASS_NAME,
                "calendar__currency"
            )

            if symbol.text in self.symbols:
                timestamp = row.find_element(
                                By.CLASS_NAME,
                                "calendar__time"
                            ).text

                impact = row.find_element(
                            By.CSS_SELECTOR,
                            "td.calendar__cell.calendar__impact span[title]"
                        ).get_attribute("title")

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
            logger.error("error getting row data -> %s", e)
            return {
                    "date": "",
                    "timestamp": "",
                    "symbol": "",
                    "impact": "",
                    "links": []
                }

    def __get_row_links(self, row: WebElement) -> list[str]:
        """
        Extracts related story links from a news row.

        Args:
            row (WebElement): The WebElement representing a row of news data.

        Returns:
            list[str]: A list of URLs (strings) pointing to related news articles.

        Behavior:
            - Clicks the detail button in the row to reveal additional information.
            - Waits for the related stories section to load.
            - Extracts links from the "flexposts__storydisplay-info" elements.

        Logs:
            - Debug messages indicating progress through the method.
            - Error message if an exception occurs during link extraction.

        Raises:
            Exception: Logs and handles errors encountered during link retrieval.
        """
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
            logger.error("error getting row links:\n %s", e)
            return []
        
    def __get_link_text(self, link: str) -> str:
        """
        Extracts the text content from a given news link.

        Args:
            link (str): The URL of the news article to extract content from.

        Returns:
            str: The extracted text content of the article. Returns an empty string 
                if the source is a YouTube link or if an error occurs.

        Behavior:
            - Opens a new browser tab and navigates to the specified link.
            - Waits for the page content to load.
            - Checks the source of the news article to ensure it's not YouTube.
            - Extracts text content from the "news__copy" class.
            - Closes the tab and switches back to the original tab.

        Logs:
            - Info message if a YouTube link is skipped.
            - Error message if an exception occurs during the text extraction process.

        Raises:
            Exception: Logs and handles errors, reinitializes the WebDriver upon failure.
        """
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
                logger.info("youtube link, skipping...")
                return ""

        except Exception as e:
            # Close the tab
            self.wd.close()

            # Switch back to the original tab
            self.wd.switch_to.window(self.wd.window_handles[0])
            logger.error("error getting link text:\n %s", e)
            self.wd.quit()
            self.__configure_webdriver()
            return ""


    @staticmethod
    def __scanDir(directory: str, extension: str, check_dir: str = None, pickup: bool = False) -> list[str]:
        """
        Scans a directory for files with a specified extension, with optional filtering based on another directory.

        Args:
            directory (str): The path to the directory to scan.
            extension (str): The file extension to filter by (e.g., ".csv").
            check_dir (str, optional): A directory to compare file names against for exclusion. Default is None.
            pickup (bool, optional): If True, filters files not already present in the `check_dir`. Default is False.

        Returns:
            list[str]: A sorted list of filenames that match the criteria.

        Behavior:
            - If `pickup` is True, only files in the source directory but not in the `check_dir` are included.
            - If `pickup` is False, all files containing "news" and having the specified extension are included.
            - The list of matching files is sorted alphabetically.

        Logs:
            - Logs the number of matching files found.

        Raises:
            None: Returns an empty list if no files match the criteria.
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
        logger.info("found %s files", len(files))
        return files
    
    @staticmethod
    def __generate_date_ranges(start_year: int, end_year: int) -> list[str]:
    def __generate_date_ranges(start_year: int, end_year: int) -> list[str]:
        """
        Generates a list of weekly date ranges between the specified start and end years.

        Args:
            start_year (int): The start year for generating the date ranges.
            end_year (int): The end year for generating the date ranges.

        Returns:
            list[str]: A list of weekly date ranges formatted as "MMMDD.YYYY-MMMDD.YYYY" (e.g., "jan01.2024-jan07.2024").

        Behavior:
            - For each year between `start_year` and `end_year`, weekly date ranges are created.
            - Each range starts on a Monday and ends on the following Sunday, ensuring the date range does not exceed the year's final date.

        Example:
            If the years are 2024 to 2025, the function will return weekly date ranges like:
            ["jan01.2024-jan07.2024", "jan08.2024-jan14.2024", ...].

        Raises:
            None: Returns an empty list if the range is invalid.
        """
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
        """
        Cleans and formats the 'timestamp' and 'date' columns in the DataFrame, 
        combining them into a single 'datetime' column.

        Args:
            df (pd.DataFrame): The DataFrame containing 'date' and 'timestamp' columns to be cleaned.

        Returns:
            pd.DataFrame: The DataFrame with an updated 'datetime' column and cleaned 'timestamp' values.

        Behavior:
            - The function first handles rows with invalid or missing timestamps by filling in the first valid timestamp.
            - Then, it iterates over the rows, filling in any missing timestamps with the previous valid timestamp.
            - The 'date' and 'timestamp' columns are combined into a single 'datetime' column, formatted as a `datetime` object.

        Example:
            Given a DataFrame with columns "date" and "timestamp", the function will add a new "datetime" column 
            with the combined and cleaned values.

        Raises:
            ValueError: If there are issues in parsing the date-time format.
        """
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
        """
        Combines multiple CSV files containing news data into a single DataFrame 
        and saves the combined data to a new CSV file.

        Args:
            source_path (str, optional): The directory path where the source CSV files are located. Default is "./data/interim".
            save_path (str, optional): The directory path where the combined CSV file will be saved. Default is "./data/interim".

        Returns:
            None: The function saves the combined DataFrame as a CSV file.

        Behavior:
            - Scans the specified directory for CSV files.
            - For each file, reads the content into a DataFrame, cleans the data, and appends it to a list.
            - Skips empty or invalid files and logs warnings for skipped files.
            - Combines all valid DataFrames into one and saves the result to a new CSV file.

        Logs:
            - Logs the number of files processed, skipped, and successfully combined.
            - Logs warnings for any issues with specific CSV files.
        """
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