"""
Example of querying nilDB with NilAI using nilRAG.
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
    Query nilDB with NilAI using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Creates a chat completion configuration
    3. Sends the query to nilAI with nilRAG
    4. Displays the response and timing information
    """
    parser = argparse.ArgumentParser(description="Query nilDB with NilAI using nilRAG")
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
    print("Loading nilDB config...")
    start_time = time.time()
    nil_db, _ = load_nil_db_config(
        args.config,
        require_bearer_token=True,
        require_schema_id=True,
        require_diff_query_id=True,
    )
    end_time = time.time()
    print(f"Config loaded successfully in {end_time - start_time:.2f} seconds")
    print()
    print(nil_db)
    print()

    print("Using prompt:", args.prompt)

    print("Querying nilAI with nilRAG...")
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
    response = nil_db.nilai_chat_completion(config)
    end_time = time.time()
    print(json.dumps(response, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
