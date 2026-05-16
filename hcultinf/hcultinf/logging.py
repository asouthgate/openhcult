from __future__ import annotations

import functools
import logging
import time


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            logger.info("%s completed in %.2f seconds", func.__qualname__, duration)

    return wrapper