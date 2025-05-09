"""
Script to write data to nilDB using nilRAG.
"""

import argparse
import asyncio
import os
import time

import numpy as np
from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault
from nilrag.utils.process import cluster_embeddings

DEFAULT_FILE_PATH = "examples/data/20-fake.txt"
DEFAULT_NUMBER_CLUSTERS = 0
DEFAULT_CHUNK_SIZE = 50


async def main():
    """
    Write data to nilDB using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Process RAG data by creating embeddings, chunks and corresponding shares
    3. Clusters embeddings if requested
    4. Writes the data to nilDB nodes
    """

    # Parser
    parser = argparse.ArgumentParser(description="Write data to nilDB using nilRAG")
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_FILE_PATH,
        help=f"Path to data file to upload (default: {DEFAULT_FILE_PATH})",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=DEFAULT_NUMBER_CLUSTERS,
        help=f"Number of clusters to use (default: {DEFAULT_NUMBER_CLUSTERS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size to use (default: {DEFAULT_CHUNK_SIZE})",
    )
    args = parser.parse_args()

    with_clustering = args.num_clusters > 1

    # Load environment variables
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Initialize vault
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        with_clustering=with_clustering,
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    )

    # Process RAG data by creating embeddings, chunks and corresponding shares
    print(f"Process RAG data...")
    start_time = time.time()
    embeddings, embeddings_shares, chunks_shares = await rag.process_rag_data(args.file, chunk_size=args.chunk_size)
    end_time = time.time()
    print(f"RAG data processed in {end_time - start_time:.2f} seconds")

    # Create clustering embeddings
    if args.num_clusters > 1:
        print("Starting clustering process:")
        print(f"    Number of embeddings: {len(embeddings)}")
        print(f"    Requested number of clusters: {args.num_clusters}")
        start_time = time.time()
        labels, centroids = cluster_embeddings(embeddings, args.num_clusters)
        print(f"Data clustered in {end_time - start_time:.2f} seconds")
        print("Cluster sizes:")
        for i in range(args.num_clusters):
            print(f"    Cluster {i}: {np.sum(labels == i)} documents")
    else:
        labels = None
        centroids = None

    # Write data
    print("Writing data...")
    start_time = time.time()
    await rag.write_rag_data(
        embeddings_shares, chunks_shares, labels=labels, centroids=centroids
    )
    end_time = time.time()
    print(f"Data written in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
