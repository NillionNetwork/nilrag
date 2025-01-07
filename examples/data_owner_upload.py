"""
Script to upload data to nilDB using nilRAG.
"""

import os
import json
import sys
import nilql
from nilrag.util import (
    create_chunks,
    encrypt_float_list,
    generate_embeddings_huggingface,
    load_file,
)
from nilrag.nildb_requests import NilDB, Node


JSON_FILE = "examples/nildb_config.json"
# Update with your secret key
SECRET_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"
FILE_PATH = 'examples/data/cities.txt'

# Load NilDB from JSON file if it exists
if os.path.exists(JSON_FILE):
    print("Loading NilDB configuration from file...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        nodes = []
        for node_data in data["nodes"]:
            nodes.append(
                Node(
                    node_data["url"],
                    node_data["node_id"],
                    node_data["org"],
                    None,
                    node_data.get("schema_id"),
                )
            )
        nilDB = NilDB(nodes)
else:
    print("Error: NilDB configuration file not found.")
    sys.exit(1)

nilDB.generate_jwt(SECRET_KEY, ttl=100000000)

print("NilDB instance:", nilDB)
print()

# Initialize secret keys for different modes of operation
num_nodes = len(nilDB.nodes)
additive_key = nilql.secret_key({'nodes': [{}] * num_nodes}, {'sum': True})
xor_key = nilql.secret_key({'nodes': [{}] * num_nodes}, {'store': True})

# Load and process input file
paragraphs = load_file(FILE_PATH)
chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)

# Generate embeddings
print('Generating embeddings...')
embeddings = generate_embeddings_huggingface(chunks)
print('Embeddings generated!')

# Encrypt chunks and embeddings
chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
embeddings_shares = [
    encrypt_float_list(additive_key, embedding) for embedding in embeddings
]

# Upload encrypted data to nilDB
print('Uploading data...')
nilDB.upload_data(embeddings_shares, chunks_shares)
