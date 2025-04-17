import time

# Benchmark
ENABLE_BENCHMARK = True  # set to False to disable


def benchmark_time(func, *args, **kwargs):
    """Measures the execution time of a sync function."""
    if ENABLE_BENCHMARK:
        start_time = time.time()
    result = func(*args, **kwargs)
    if ENABLE_BENCHMARK:
        return result, time.time() - start_time
    return result, None


async def benchmark_time_async(func, *args, **kwargs):
    """Measures the execution time of an async function."""
    if ENABLE_BENCHMARK:
        start_time = time.time()
    result = await func(*args, **kwargs)
    if ENABLE_BENCHMARK:
        return result, time.time() - start_time
    return result, None
