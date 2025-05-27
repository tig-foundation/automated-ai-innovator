import logging
import sys


def setup_logger(logger_name: str, logger_filename: str, logging_level: int = logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)  # Set the level to capture all types of logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    # create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if logger_filename is not None:  # create the file handler to the logger
        file_handler = logging.FileHandler(logger_filename)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    logger.propagate = False

    return logger