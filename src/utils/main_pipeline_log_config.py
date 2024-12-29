"""
Configuration module for main pipeline logs
"""

import logging
from logging import Logger


def logger_main_pipeline() -> Logger:
    """Format logger, configure file handler and add handler
    for main pipeline logger.

    Returns:
        Logger: Logger for main pipeline logs
    """

    main_pipeline_logger = logging.getLogger(__name__)
    main_pipeline_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s:'
    )

    main_pipeline_file_handler = logging.FileHandler('./logs/main_pipeline.log')
    main_pipeline_file_handler.setLevel(logging.DEBUG)
    main_pipeline_file_handler.setFormatter(formatter)

    main_pipeline_logger.addHandler(main_pipeline_file_handler)

    return main_pipeline_logger

logger = logger_main_pipeline()