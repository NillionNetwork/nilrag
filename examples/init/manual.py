"""
Script to initialize schema and query to nilDB manually.
"""

import argparse
import asyncio
import os

from dotenv import set_key

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault


def parse_arguments():
    parser = argparse.ArgumentParser(description="Manual initialization script.")
    parser.add_argument('--env-path', type=str, default=".env", help='Path to the environment file.')
    parser.add_argument('--with-clustering', type=bool, default=True, help='Enable clustering.')
    return parser.parse_args()

args = parse_arguments()
ENV_PATH = args.env_path
WITH_CLUSTERING = args.with_clustering


async def main():
    """
    1. Loads nilDB config
    2. Creates schemas and query
    3. Updates .env with generated IDs
    """
    # Initialize vault with clustering enabled
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"], ORG_CONFIG["org_credentials"], with_clustering=WITH_CLUSTERING
    )

    # Write them back into the .env file
    if not os.path.exists(ENV_PATH):
        raise FileNotFoundError(f"{ENV_PATH} not found")

    # Create schemas & query
    schema_id = await rag.create_rag_schema()
    subtract_query_id = await rag.create_subtract_query()
    clusters_schema_id = (
        await rag.create_clusters_schema() if WITH_CLUSTERING else None
    )

    updates = {
        "SCHEMA_ID": schema_id,
        "CLUSTERS_SCHEMA_ID": clusters_schema_id if WITH_CLUSTERING else "",
        "QUERY_ID": subtract_query_id,
    }

    for key, value in updates.items():
        set_key(ENV_PATH, key, value)

    print("✅ Updated .env with:")
    print(f"   SCHEMA_ID={schema_id}")
    print(f"   CLUSTERS_SCHEMA_ID={clusters_schema_id}")
    print(f"   QUERY_ID={subtract_query_id}")


if __name__ == "__main__":
    asyncio.run(main())
