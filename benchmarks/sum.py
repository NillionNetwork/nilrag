import pytest

def test_fib_10(benchmark):
    benchmark.pedantic(lambda: 10, iterations=10, rounds=100)