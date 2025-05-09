"""
Example of querying NilAI endpoint using nilRAG.
"""

import argparse
import asyncio
import json
import os
import time

from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import ChatCompletionConfig, RAGVault

DEFAULT_PROMPT = "Who is Michelle Ross?"


async def main():
    """
    Query NilAI endpoint using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Creates a chat completion configuration
    3. Sends the query to nilAI with nilRAG
    4. Displays the response and timing information
    """
    parser = argparse.ArgumentParser(description="Query NilAI endpoint using nilRAG")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Query prompt (default: {DEFAULT_PROMPT})",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Initialize vault with clustering enabled
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    )

    print("Query nilAI with nilRAG...")
    start_time = time.time()
    config = ChatCompletionConfig(
        nilai_url="https://nilai-a779.nillion.network",
        token="Nillion2025",
        messages=[{"role": "user", "content": args.prompt}],
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.2,
        max_tokens=2048,
        stream=False,
    )
    response = rag.nilai_chat_completion(config)
    end_time = time.time()
    print(json.dumps(response, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
