import logging
import time


def get_logger(module_name: str) -> logging.Logger:
    return logging.getLogger("uvicorn.error").getChild(__name__)


def logged_timed_func_call(logger, func, *args, **kwargs):
    """
    Helper function to log the execution time of a synchronous function call.
    """
    start_time = time.time()
    try:
        return func(*args, **kwargs)
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Executed {func.__name__} in {elapsed_time:.2f} seconds")


def timed_func(func):
    """
    Decorator for synchronous functions to log execution time.
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        return logged_timed_func_call(logger, func, *args, **kwargs)

    return wrapper
