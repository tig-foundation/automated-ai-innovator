"""
Utility functions for scripting
"""

import importlib
import logging
import os



def setup_logger(logger_name: str, logger_filename: str, logging_level: int = logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)  # Set the level to capture all types of logs

    # create the file handler to the logger
    file_handler = logging.FileHandler(logger_filename)
    file_handler.setLevel(logging_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def import_model_specific_symbols(file_name: str, symbol_names: list[str]):
    """
    import class from Python script given by a filename
    """
    module_name = os.path.splitext(file_name)[0]
    if os.path.isfile(file_name):
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            my_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(my_module)
            
            symbol_list = []
            for symbol_name in symbol_names:
                symbol_obj = getattr(my_module, symbol_name, None)
                if symbol_obj is None:
                    raise ValueError(f"{symbol_name} not found in {file_name}.")
                symbol_list.append(symbol_obj)

            return symbol_list

        except Exception as e:
            raise ValueError(f"Error importing attribute: {e}")
    else:
        raise ValueError(f"The file {file_name} does not exist.")
    
    
def get_nested_attr(obj, attr_path):
    """
    Recursively access nested attributes using a dot-separated string.
    """
    attrs = attr_path.split(".")  # e.g. split "layer1.weight" into ["layer1", "weight"]
    for attr in attrs:
        obj = getattr(obj, attr)  # navigate deeper
    return obj


def set_nested_attr(obj, attr_path, value):
    """
    Recursively set a nested attribute using a dot-separated string.
    """
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:  # traverse up to the last key
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)  # update the final attribute