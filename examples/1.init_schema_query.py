"""
Script to initialize schema and query to nilDB.
"""

import argparse
import asyncio
import json
import time

from nilrag.config import load_nil_db_config

DEFAULT_CONFIG = "examples/nildb_config.json"


async def main():
    """
    Initialize schema and query for nilDB nodes.

    This script:
    1. Loads the nilDB configuration from a JSON file
    2. Generates JWT tokens for authentication
    3. Creates a schema for storing embeddings and chunks
    4. Creates a query for computing differences between embeddings
    5. Updates the configuration file with the generated IDs and tokens
    """
    parser = argparse.ArgumentParser(
        description="Initialize schema and query for nilDB"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to nilDB config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    # Load NilDB configuration
    nil_db, secret_key = load_nil_db_config(
        args.config, require_secret_key=True
    )
    jwts = nil_db.generate_jwt(secret_key, ttl=3600)
    print(nil_db)
    print()

    # Upload encrypted data to nilDB
    print("Initializing schema...")
    start_time = time.time()
    schema_id = await nil_db.init_schema()
    end_time = time.time()
    print(f"Schema initialized successfully in {end_time - start_time:.2f} seconds")

    print("Initializing query...")
    start_time = time.time()
    diff_query_id = await nil_db.init_diff_query()
    end_time = time.time()
    print(f"Query initialized successfully in {end_time - start_time:.2f} seconds")

    # Update config file with new IDs and tokens
    with open(args.config, "r", encoding="utf-8") as f:
        data = json.load(f)
    for node_data, jwt in zip(data["nodes"], jwts):
        node_data["schema_id"] = schema_id
        node_data["diff_query_id"] = diff_query_id
        node_data["bearer_token"] = jwt
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Updated nilDB configuration file with schema and query IDs.")


if __name__ == "__main__":
    asyncio.run(main())
