"""
Example of querying nilDB with NilAI using nilRAG.
"""

import argparse
import asyncio
import json
import time

from nilrag.config import load_nil_db_config
from nilrag.nildb_requests import ChatCompletionConfig

DEFAULT_CONFIG = "examples/nildb_config.json"
DEFAULT_PROMPT = "Who is Danielle Miller?"
DEFAULT_FAKE_QUERIES = 0


async def main():
    """
    Query nilDB with NilAI using nilRAG.

    This script:
    1. Loads the nilDB configuration
    2. Creates a chat completion configuration
    3. Sends the query to nilAI with nilRAG
    4. Displays the response and timing information
    """
    parser = argparse.ArgumentParser(description="Query nilDB with NilAI using nilRAG")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to nilDB config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Query prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--fake_queries",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Number of fake queries to produce (default: {DEFAULT_FAKE_QUERIES})",
    )
    args = parser.parse_args()

    # Load NilDB configuration
    nil_db, _ = load_nil_db_config(
        args.config,
        require_bearer_token=True,
        require_schema_id=True,
        require_diff_query_id=True,
        require_clusters_schema_id=True,
    )
    print(nil_db)
    print()

    # Check if clustering was performed
    print("Starting cluster check...")
    start_time = time.time()
    num_clusters = await nil_db.check_clustering()
    cluster_check_time = time.time() - start_time
    print(f"Cluster check result: {num_clusters} clusters found")
    print(f"Cluster check took {cluster_check_time:.2f} seconds")
    
    filter = {}
    closest_centroid = None  # NEW: Variable to store the closest centroid
    if num_clusters > 1:
        print(f"Clustering was performed - found {num_clusters} clusters")
        # Find closest centroid using the prompt
        print(f"Finding closest centroid for prompt: {args.prompt}")
        start_time = time.time()
        closest_centroid_idx, closest_centroid, centroid_id = await nil_db.get_closest_centroid_index(args.prompt)
        centroid_time = time.time() - start_time
        print(f"Finding closest centroid took {centroid_time:.2f} seconds")
        
        if closest_centroid_idx is not None:
            print(f"Closest centroid found:")
            print(f"Index: {closest_centroid_idx}")
            print(f"ID: {centroid_id}")
            print(f"Centroid vector (first 5 values): {closest_centroid[:5]}")
            filter = {
                "cluster_centroid": closest_centroid
            }
    else:
        print("No clustering was performed - no centroids found in clusters schema")

    print("Query nilAI with nilRAG...")
    start_time = time.time()
    config = ChatCompletionConfig(
        nilai_url="https://nilai-a779.nillion.network",
        token="Nillion2025",
        messages=[{"role": "user", "content": args.prompt}],
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.2,
        max_tokens=2048,
        stream=False,
        filter=filter,
    )
    #Pass closest_centroid to diff_query_execute
    response = nil_db.nilai_chat_completion(config, closest_centroid=closest_centroid)
    query_time = time.time() - start_time
    print(json.dumps(response, indent=4))
    print(f"Timing summary:")
    if num_clusters > 1:
        print(f"Cluster check: {cluster_check_time:.2f} seconds")
        print(f"Finding closest centroid: {centroid_time:.2f} seconds")
    print(f"Query execution: {query_time:.2f} seconds")
    print(f"Total time: {cluster_check_time + (centroid_time if num_clusters > 1 else 0) + query_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
