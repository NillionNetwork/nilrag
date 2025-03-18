"""
Script to upload data to nilDB using nilRAG.
"""

import os
import json
import sys
import argparse
import time
import nilql
from nilrag.util import (
    create_chunks,
    encrypt_float_list,
    generate_embeddings_huggingface,
    load_file,
)
from nilrag.nildb_requests import NilDB, Node


DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_FILE_PATH = 'examples/data/20-fake.txt'

def main():
    parser = argparse.ArgumentParser(description='Upload data to nilDB using nilRAG')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                      help=f'Path to nilDB config file (default: {DEFAULT_CONFIG})')
    parser.add_argument('--file', type=str, default=DEFAULT_FILE_PATH,
                      help=f'Path to data file to upload (default: {DEFAULT_FILE_PATH})')
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
                        node_data["url"],
                        node_data["node_id"],
                        data["org_did"],
                        node_data["bearer_token"],
                        node_data.get("schema_id"),
                    )
                )
            nilDB = NilDB(nodes)
    else:
        print(f"Error: NilDB configuration file not found at {args.config}")
        sys.exit(1)

    print(nilDB)
    print()

    # Initialize secret keys for different modes of operation
    num_nodes = len(nilDB.nodes)
    additive_key = nilql.ClusterKey.generate({'nodes': [{}] * num_nodes}, {'sum': True})
    xor_key = nilql.ClusterKey.generate({'nodes': [{}] * num_nodes}, {'store': True})

    # Load and process input file
    paragraphs = load_file(args.file)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)

    # Generate embeddings and chunks
    print('Generating embeddings and chunks...')
    start_time = time.time()
    embeddings = generate_embeddings_huggingface(chunks)
    end_time = time.time()
    print(f'Embeddings and chunks generated in {end_time - start_time:.2f} seconds!')

    # Encrypt chunks and embeddings
    print('Encrypting data...')
    start_time = time.time()
    chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
    embeddings_shares = [
        encrypt_float_list(additive_key, embedding) for embedding in embeddings
    ]
    end_time = time.time()
    print(f'Data encrypted in {end_time - start_time:.2f} seconds')

    # Upload encrypted data to nilDB
    print('Uploading data...')
    start_time = time.time()
    nilDB.upload_data(embeddings_shares, chunks_shares)
    end_time = time.time()
    print(f'Data uploaded in {end_time - start_time:.2f} seconds')

if __name__ == "__main__":
    main()
