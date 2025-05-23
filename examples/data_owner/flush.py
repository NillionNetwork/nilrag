"""
Script to clear data from all nilDB nodes.
"""

import asyncio
import os

from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault


async def main():
    """
    Write data to nilDB using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Flushes data
    """

    # Load environment variables
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    # Sets clustering to True if clusters_schema_id is not None or not empty
    with_clustering = bool(clusters_schema_id)

    # Initialize vault with clustering enabled
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        with_clustering=with_clustering,
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
    )

    # Delete chunk and embedding data
    await rag.flush_data()
    # Delete clusters data in case clustering is enabled
    if with_clustering:
        rag.schema_id = clusters_schema_id
        await rag.flush_data()


if __name__ == "__main__":
    asyncio.run(main())
