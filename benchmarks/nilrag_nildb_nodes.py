"""
Benchmarks of server performing RAG with nilDB nodes.
"""

import argparse
import asyncio
import json
import os
import time

from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault

DEFAULT_PROMPT = "Who is Michelle Ross?"
DEFAULT_NUM_CHUNKS = 2
DEFAULT_NUM_CLUSTERS = 1
ENABLE_BENCHMARKS = True


async def main():
    """
    Performing RAG using nilDB nodes. This is the RAG logic to be run on nilAI.

    This script:
    1. Loads the nilDB configuration
    2. Performs RAG using nilDB nodes
    3. Displays the response and timing information
    """
    parser = argparse.ArgumentParser(description="Query nilDB using nilRAG")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Query prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "-c",
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of chunks to return (default: {DEFAULT_NUM_CHUNKS})",
    )
    parser.add_argument(
        "-l",
        "--num-clusters",
        type=int,
        default=DEFAULT_NUM_CLUSTERS,
        help=f"Number of clusters to search through (default: {DEFAULT_NUM_CLUSTERS})",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Initialize vault
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    )

    print("Perform nilRAG...")
    start_time = time.time()
    top_chunks = await rag.top_num_chunks_execute(
        args.prompt, args.num_chunks, ENABLE_BENCHMARKS, args.num_clusters
    )
    end_time = time.time()
    print(json.dumps(top_chunks, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
