"""
Configuration module for model training logs
"""

import logging
from logging import Logger


def logger_model_train() -> Logger:
    """Format logger, configure file handler and add handler
    for model training logger.

    Returns:
        Logger: Logger for model training logs
    """

    model_train_logger = logging.getLogger(__name__)
    model_train_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:'
    )

    model_train_file_handler = logging.FileHandler('./logs/model_train.log')
    model_train_file_handler.setLevel(logging.DEBUG)
    model_train_file_handler.setFormatter(formatter)

    model_train_logger.addHandler(model_train_file_handler)

    return model_train_logger

logger = logger_model_train()
