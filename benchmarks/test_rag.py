import asyncio
import os
from dotenv import load_dotenv
import pytest
import time


from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault

def test_rag_pedantic(benchmark):
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Setup RAG instance
    rag = asyncio.run(RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    ))

    prompt = "Who is Michelle Ross?"
    num_chunks = 2
    num_clusters = 1

    def sync_runner():

        return asyncio.run(rag.top_num_chunks_execute(
            prompt, num_chunks, True, num_clusters
        ))
    
    # Warm up
    for _ in range(5):
        sync_runner()

    #Actual benchmark
    result = benchmark.pedantic(sync_runner, iterations=5, rounds=3)

    assert isinstance(result, list)
