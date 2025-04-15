"""
Example of server performing RAG with nilDB nodes.
"""

import argparse
import asyncio
import json
import time

from nilrag.config import load_nil_db_config
from nilrag.nildb_requests import ChatCompletionConfig

DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_PROMPT = "Who is Danielle Miller?"


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
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to nilDB config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Query prompt (default: {DEFAULT_PROMPT})",
    )
    args = parser.parse_args()

    # Load NilDB configuration
    nil_db, _ = load_nil_db_config(
        args.config,
        require_bearer_token=True,
        require_schema_id=True,
        require_diff_query_id=True,
        require_clusters_schema_id=True,
        require_cluster_diff_query_id=True,
    )
    print(nil_db)
    print()

    print("Perform nilRAG...")
    start_time = time.time()
    top_chunks = await nil_db.top_num_chunks_execute(args.prompt, 2)
    end_time = time.time()
    print(json.dumps(top_chunks, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
