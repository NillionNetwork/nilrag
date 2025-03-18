"""
Script to initialize schema and query to nilDB.
"""

import os
import json
import sys
import argparse
import time
import asyncio
from nilrag.nildb_requests import NilDB, Node


DEFAULT_CONFIG = "examples/nildb_config.json"

async def main():
    parser = argparse.ArgumentParser(description='Initialize schema and query for nilDB')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                      help=f'Path to nilDB config file (default: {DEFAULT_CONFIG})')
    args = parser.parse_args()

    # Load NilDB from JSON file if it exists
    secret_key = None
    if os.path.exists(args.config):
        print(f"Loading NilDB configuration from {args.config}...")
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
            secret_key = data["org_secret_key"]
            nodes = []
            for node_data in data["nodes"]:
                nodes.append(
                    Node(
                        node_data["url"],
                        node_data["node_id"],
                        data["org_did"],
                    )
                )
            nilDB = NilDB(nodes)
    else:
        print(f"Error: NilDB configuration file not found at {args.config}")
        sys.exit(1)

    jwts = nilDB.generate_jwt(secret_key, ttl=3600)

    print(nilDB)
    print()

    # Upload encrypted data to nilDB
    print("Initializing schema...")
    start_time = time.time()
    schema_id = await nilDB.init_schema()
    end_time = time.time()
    print(f"Schema initialized successfully in {end_time - start_time:.2f} seconds")

    print("Initializing query...")
    start_time = time.time()
    diff_query_id = await nilDB.init_diff_query()
    end_time = time.time()
    print(f"Query initialized successfully in {end_time - start_time:.2f} seconds")

    with open(args.config, "w", encoding="utf-8") as f:
        for node_data, jwt in zip(data["nodes"], jwts):
            node_data["schema_id"] = schema_id
            node_data["diff_query_id"] = diff_query_id
            node_data["bearer_token"] = jwt
        json.dump(data, f, indent=4)
    print("Updated nilDB configuration file with schema and query IDs.")

if __name__ == "__main__":
    asyncio.run(main())
