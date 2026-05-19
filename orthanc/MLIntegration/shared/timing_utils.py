"""
Timing utilities for performance instrumentation
"""
import time
import logging
from contextlib import contextmanager
from typing import Generator


@contextmanager
def time_operation(operation_name: str, logger: logging.Logger) -> Generator[None, None, None]:
    """
    Context manager for timing operations and logging results

    Usage:
        with time_operation("model_inference", logger):
            result = model.predict(data)

    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance to use for output

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"TIMING: {operation_name}: {duration_ms:.2f}ms")
