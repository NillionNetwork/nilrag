"""
Benchmarks for search quality with nilDB nodes.
"""

import argparse
import asyncio
import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from faker import Faker

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault
from nilrag.utils.process import generate_embeddings_huggingface

DEFAULT_NUM_CHUNKS = 10
DEFAULT_NUM_CLUSTERS = 10
DEFAULT_NUM_QUERIES = 10
ENABLE_BENCHMARKS = True


def precision_at_k(pred, truth):
    assert len(pred) == len(truth)
    k = len(pred)
    precision = 0
    for i in range(k):
        precision += pred[i] in truth
    return precision / k


def normalized_discounted_cumulative_gain(pred, truth):
    assert len(pred) == len(truth)
    k = len(pred)
    dcg, ideal_dcg = 0, 0
    for i in range(k):
        ideal_dcg += (k - i) / np.log2(i + 2)
        for j in range(k):
            if pred[i] == truth[j]:
                dcg += (k - j) / np.log2(i + 2)
                break
    return dcg / ideal_dcg


async def main():
    """
    Performing search using nilDB nodes. This is the search logic to be run on nilAI.

    This script:
    1. Loads the nilDB configuration
    2. Fetches the number of clusters to use for full search
    3. Generates random queries
    4. For each query performs full search
    5. For each query performs partial search using clusters
    6. Displays the response and timing information
    """
    parser = argparse.ArgumentParser(description="Query nilDB using nilRAG")
    parser.add_argument(
        "-c",
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of chunks to return (default: {DEFAULT_NUM_CHUNKS})",
    )
    parser.add_argument(
        "-l",
        "--num-clusters",
        type=int,
        default=DEFAULT_NUM_CLUSTERS,
        help=f"Number of clusters to search through (default: {DEFAULT_NUM_CLUSTERS})",
    )
    parser.add_argument(
        "-q",
        "--num-queries",
        type=int,
        default=DEFAULT_NUM_QUERIES,
        help=f"The number of queries to test (default: {DEFAULT_NUM_QUERIES}).",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")

    # Initialize vault
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    )

    fake = Faker()

    print("Get clusters...")
    embedding = generate_embeddings_huggingface("")
    num_clusters, _ = await rag.get_closest_centroids(embedding)

    print("Perform nilRAG...")
    start_times, end_times = [], []
    precisions, ndcgs = [], []
    for i in range(args.num_queries):
        prompt = f"Who is {fake.name()}?"
        print(f"Prompt {i}: {prompt}")
        top_chunks = await rag.top_num_chunks_execute(
            prompt, args.num_chunks, ENABLE_BENCHMARKS, num_clusters
        )
        print(json.dumps(top_chunks, indent=4))
        start_times.append(time.time())
        selected_chunks = await rag.top_num_chunks_execute(
            prompt, args.num_chunks, ENABLE_BENCHMARKS, args.num_clusters
        )
        end_times.append(time.time())
        print(json.dumps(selected_chunks, indent=4))
        precisions.append(precision_at_k(selected_chunks, top_chunks))
        ndcgs.append(normalized_discounted_cumulative_gain(selected_chunks, top_chunks))
    running_times = np.array(end_times) - np.array(start_times)
    print(
        f"Queries took {np.mean(running_times):.2f} +- {np.std(running_times):.2f} seconds"
    )
    print(f"Precision {np.mean(precisions):.2f} +- {np.std(precisions):.2f}")
    print(f"nDCG {np.mean(ndcgs):.2f} +- {np.std(ndcgs):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
