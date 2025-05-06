"""
This module provides functions to benchmark the execution time of functions.
"""

import time


def benchmark_time(func, *args, enable=False, **kwargs):
    """Measures the execution time of a sync function."""
    if enable:
        start_time = time.time()
    result = func(*args, **kwargs)
    if enable:
        return result, time.time() - start_time
    return result, None


async def benchmark_time_async(func, *args, enable=False, **kwargs):
    """Measures the execution time of an async function."""
    if enable:
        start_time = time.time()
    result = await func(*args, **kwargs)
    if enable:
        return result, time.time() - start_time
    return result, None
