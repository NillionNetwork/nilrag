"""
Benchmarks for RAG performance using pytest-benchmark.

This script:
1. Sets up a RAG instance with nilDB configuration
2. Performs a warm-up phase to ensure stable measurements
3. Runs benchmark tests using pytest-benchmark's pedantic mode
4. Measures execution time and performance metrics for RAG operations.
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault


def test_rag_pedantic(benchmark):
    """
    Benchmark test for RAG performance using pytest-benchmark's pedantic mode.

    This test:
    1. Initializes the RAG system with nilDB configuration
    2. Sets up test parameters (prompt, number of chunks, clusters)
    3. Performs a warm-up phase to stabilize measurements
    4. Runs the benchmark with multiple iterations and rounds
    5. Verifies the result is a list

    Args:
        benchmark: pytest-benchmark fixture for performance testing
    """
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Setup RAG instance
    rag = asyncio.run(
        RAGVault.create(
            ORG_CONFIG["nodes"],
            ORG_CONFIG["org_credentials"],
            schema_id=schema_id,
            clusters_schema_id=clusters_schema_id,
            subtract_query_id=subtract_query_id,
        )
    )

    prompt = "Who is Michelle Ross?"
    num_chunks = 2
    num_clusters = 1

    def sync_runner():
        """
        Synchronous wrapper for the async RAG execution.

        This function:
        1. Wraps the async top_num_chunks_execute in a synchronous context
        2. Executes the RAG query with the configured parameters
        3. Returns the retrieved chunks

        Returns:
            list: Retrieved chunks from the RAG system
        """
        return asyncio.run(
            rag.top_num_chunks_execute(prompt, num_chunks, False, num_clusters)
        )

    # Warm up
    for _ in range(10):
        sync_runner()
    # Actual benchmark
    result = benchmark.pedantic(sync_runner, iterations=10, rounds=5)

    assert isinstance(result, str)
