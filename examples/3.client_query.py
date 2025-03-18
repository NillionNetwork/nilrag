"""
Example of querying nilDB with NilAI using nilRAG.
"""

import os
import sys
import json
import argparse
import time
import asyncio
from nilrag.nildb_requests import NilDB, Node


DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_PROMPT = "Who is Danielle Miller?"

async def main():
    parser = argparse.ArgumentParser(description='Query nilDB with NilAI using nilRAG')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                      help=f'Path to nilDB config file (default: {DEFAULT_CONFIG})')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                      help=f'Query prompt (default: {DEFAULT_PROMPT})')
    args = parser.parse_args()

    # Load NilDB from JSON file if it exists
    if os.path.exists(args.config):
        print(f"Loading NilDB configuration from {args.config}...")
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
            nodes = []
            for node_data in data["nodes"]:
                nodes.append(
                    Node(
                        url=node_data["url"],
                        node_id=node_data.get("node_id"),
                        org=data.get("org_did"),
                        bearer_token=node_data.get("bearer_token"),
                        schema_id=node_data.get("schema_id"),
                        diff_query_id=node_data.get("diff_query_id"),
                    )
                )
            nilDB = NilDB(nodes)
    else:
        print(f"Error: NilDB configuration file not found at {args.config}")
        sys.exit(1)

    print(nilDB)
    print()

    print('Query nilAI with nilRAG...')
    start_time = time.time()
    response = nilDB.nilai_chat_completion(
        nilai_url="https://nilai-a779.nillion.network",
        token="Nillion2025",
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": args.prompt}
        ],
        temperature=0.2,
        max_tokens=2048,
        stream=False,
    )
    end_time = time.time()
    print(json.dumps(response, indent=4))
    print(f"Query took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
