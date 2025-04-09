"""
Script to upload data to nilDB using nilRAG.
"""

import argparse
import asyncio
import time

import nilql

from nilrag.config import load_nil_db_config
from nilrag.util import (create_chunks, encrypt_float_list,
                         generate_embeddings_huggingface, load_file,
                         cluster_embeddings)

DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_FILE_PATH = "examples/data/20-fake.txt"
DEFAULT_NUMBER_CLUSTERS = 1


async def main():
    """
    Upload data to nilDB using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Initializes encryption keys for different modes
    3. Processes the input file into chunks and embeddings
    4. Encrypts the data using nilQL
    5. Uploads the encrypted data to nilDB nodes
    """
    parser = argparse.ArgumentParser(description="Upload data to nilDB using nilRAG")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to nilDB config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_FILE_PATH,
        help=f"Path to data file to upload (default: {DEFAULT_FILE_PATH})",
    )
    parser.add_argument(
    "--clusters",
    type=int,
    default=1,
    help="Number of clusters to use (default: {DEFAULT_NUMBER_CLUSTERS})"
    )
    args = parser.parse_args()

    # Load NilDB configuration
    nil_db, _ = load_nil_db_config(
        args.config,
        require_bearer_token=True,
        require_schema_id=True,
        require_clusters_schema_id=True,
    )
    print(nil_db)
    print()

    # Initialize secret keys for different modes of operation
    num_nodes = len(nil_db.nodes)
    additive_key = nilql.ClusterKey.generate({"nodes": [{}] * num_nodes}, {"sum": True})
    xor_key = nilql.ClusterKey.generate({"nodes": [{}] * num_nodes}, {"store": True})

    # Load and process input file
    paragraphs = load_file(args.file)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)

    # Generate embeddings and chunks
    print("Generating embeddings and chunks...")
    start_time = time.time()
    embeddings = generate_embeddings_huggingface(chunks)
    end_time = time.time()
    print(f"Embeddings and chunks generated in {end_time - start_time:.2f} seconds!")

    # Encrypt chunks and embeddings
    print("Encrypting data...")
    start_time = time.time()
    chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
    embeddings_shares = [
        encrypt_float_list(additive_key, embedding) for embedding in embeddings
    ]
    end_time = time.time()
    print(f"Data encrypted in {end_time - start_time:.2f} seconds")

    # Upload encrypted data to nilDB
    if args.clusters > 1:
        # Clustering embeddings
        print(f"Clustering the data in {args.clusters} clusters...")
        start_time = time.time()
        labels, centroids = cluster_embeddings(embeddings, args.clusters)
        end_time = time.time()
        print(f"Data clustered in {end_time - start_time:.2f} seconds")
        print("Uploading data with clustering labels...")
        start_time = time.time()
        await nil_db.upload_data(embeddings_shares, chunks_shares, labels = labels, centroids = centroids)
        end_time = time.time()
        print(f"Data uploaded in {end_time - start_time:.2f} seconds")
    else :
        print("Uploading data...")
        start_time = time.time()
        await nil_db.upload_data(embeddings_shares, chunks_shares)
        end_time = time.time()
        print(f"Data uploaded in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
