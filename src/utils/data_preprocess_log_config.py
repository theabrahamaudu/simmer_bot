"""
Configuration module for data fetching logs
"""

import logging
from logging import Logger


def logger_data_preprocess() -> Logger:
    """Format logger, configure file handler and add handler
    for data preprocessing logger.

    Returns:
        Logger: Logger for data preprocessing logs
    """

    data_preprocess_logger = logging.getLogger(__name__)
    data_preprocess_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:'
    )

    data_preprocess_file_handler = logging.FileHandler('./logs/data_preprocess.log')
    data_preprocess_file_handler.setLevel(logging.DEBUG)
    data_preprocess_file_handler.setFormatter(formatter)

    data_preprocess_logger.addHandler(data_preprocess_file_handler)

    return data_preprocess_logger

logger = logger_data_preprocess()
