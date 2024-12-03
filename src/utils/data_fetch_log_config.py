"""
Configuration module for data fetching logs
"""

import logging
from logging import Logger


def logger_data_fetch() -> Logger:
    """Format logger, configure file handler and add handler
    for data fetching logger.

    Returns:
        Logger: Logger for data fetching logs
    """

    data_fetch_logger = logging.getLogger(__name__)
    data_fetch_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:'
    )

    data_fetch_file_handler = logging.FileHandler('./logs/data_fetch.log')
    data_fetch_file_handler.setLevel(logging.DEBUG)
    data_fetch_file_handler.setFormatter(formatter)

    data_fetch_logger.addHandler(data_fetch_file_handler)

    return data_fetch_logger

logger = logger_data_fetch()
