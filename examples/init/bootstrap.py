"""
Script to initialize schema and query to nilDB.
"""

import argparse
import asyncio

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bootstrap initialization script.")
    parser.add_argument(
        "--env-path",
        type=str,
        default=".env",
        help="Path to the environment file.",
    )
    parser.add_argument(
        "--with-clustering",
        action="store_true",
        help="Enable clustering (default: off).",
    )
    return parser.parse_args()

args = parse_arguments()
ENV_PATH = args.env_path
WITH_CLUSTERING = args.with_clustering


async def main():
    # run bootstrap and capture all outputs
    rag, schema_id, clusters_schema_id, query_id = await RAGVault.bootstrap(
        ORG_CONFIG, with_clustering=WITH_CLUSTERING, env_path=ENV_PATH
    )

    print("✅ Bootstrapped RAGVault and updated .env:")
    print(f"   NILLION_ORG_DID       = {ORG_CONFIG['org_credentials']['org_did']}")
    print(f"   NILLION_ORG_SECRET_KEY= <hidden>")
    print(f"   SCHEMA_ID             = {schema_id}")
    print(f"   CLUSTERS_SCHEMA_ID    = {clusters_schema_id}")
    print(f"   QUERY_ID              = {query_id}")
    for idx, node in enumerate(ORG_CONFIG["nodes"], start=1):
        print(f"   URL{idx}               = {node['url']}")
        print(f"   DID{idx}               = {node['did']}")


if __name__ == "__main__":
    asyncio.run(main())
