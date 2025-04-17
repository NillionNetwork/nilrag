import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import nilql
import numpy as np
import requests
from secretvaults import SecretVaultWrapper

from .nildb.initialization import NilDBInit
from .nildb.operations import NilDBOps
from .utils.benchmark import (ENABLE_BENCHMARK, benchmark_time,
                              benchmark_time_async)
from .utils.process import generate_embeddings_huggingface
from .utils.transform import (decrypt_float_list, encrypt_float_list,
                              group_shares_by_id, to_fixed_point)


@dataclass
class ChatCompletionConfig:
    """Configuration for chat completion requests."""

    nilai_url: str
    token: str
    messages: list[dict]
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class RAGVault(SecretVaultWrapper, NilDBInit, NilDBOps):

    def __init__(
        self,
        with_clustering: bool = False,
        clusters_schema_id: str = None,
        subtract_query_id: str = None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.clusters_schema_id = clusters_schema_id
        self.with_clustering = with_clustering
        self.subtract_query_id = subtract_query_id

    async def get_closest_centroid(
        self, query_embedding: np.ndarray
    ) -> tuple[int, Optional[List[float]]]:
        """
        Check if clustering was performed and return the number of clusters.

        Returns:
            int: Number of clusters found (0 if no clustering was performed)
        """

        try:
            # Read from first node
            node = self.nodes[0]
            schema_id = self.clusters_schema_id
            data_filter = {}
            clusters_data = await self.read_from_node(node, schema_id, data_filter)
            if not clusters_data:
                # No clusters found
                return 0, None
            num_clusters = len(clusters_data)
            centroids = [centroid["cluster_centroid"] for centroid in clusters_data]
            closest_centroid = compute_closest_centroid(query_embedding, centroids)
            return num_clusters, closest_centroid
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
            print(f"Error checking clusters and finding closest centroid: {str(e)}")
            return 0, None

    # pylint: disable=too-many-locals
    async def top_num_chunks_execute(self, query: str, num_chunks: int) -> List:
        """
        Retrieves the top `num_chunks` most relevant data chunks for a given query.

        It performs the following steps:
        1. Generates embeddings for the input query.
        2. Encrypts the query embeddings.
        3. Computes the difference between the encrypted query embeddings and stored data
        embeddings.
        4. Decrypts the differences to compute distances.
        5. Identifies and retrieves the top `num_chunks` data chunks that are most relevant to
        the query.

        Args:
            query (str): The input query string for which relevant data chunks are to be retrieved.
            num_chunks (int): The number of top relevant data chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - `_id` (Any): The unique identifier of the data chunk.
                - `distances` (float): The computed distance between the query and the data chunk.
        """
        # Check input format
        if query is None and not isinstance(query, str):
            raise TypeError("Prompt must be a string")

        # Initialize secret keys
        num_parties = len(self.nodes)
        additive_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"sum": True}
        )
        xor_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"store": True}
        )
        # Step 1: Encrypt query embedding and get closest centroid
        # Note: one string query is assumed.
        # 1.1 Generate query embedding
        query_embedding, generate_query_embedding_time_sec = benchmark_time(
            generate_embeddings_huggingface, query
        )

        # 1.2 Check if clustering was performed and (if existing) get closest centroid
        (
            num_clusters,
            closest_centroid,
        ), cluster_check_and_get_closest_centroid_time_sec = await benchmark_time_async(
            self.get_closest_centroid, query_embedding
        )
        if num_clusters > 1 and closest_centroid is not None:
            print(
                f"Clustering was performed: found {num_clusters} clusters\
                and closest centroid to query"
            )
        else:
            print("No clustering was performed")
        # 1.3 Encrypt query embedding
        nilql_query_embedding, encrypt_query_embedding_time_sec = benchmark_time(
            encrypt_float_list, additive_key, query_embedding
        )

        # Step 2: Ask NilDB to compute the differences
        difference_shares, asking_nildb_time_sec = await benchmark_time_async(
            self.diff_query_execute, nilql_query_embedding, closest_centroid
        )

        # Step 3: Compute distances and sort
        # 3.1 Group difference shares by ID
        difference_shares_by_id, group_shares_by_id_time_sec = benchmark_time(
            group_shares_by_id, difference_shares, lambda share: share["difference"]
        )
        # 3.2 Transpose the lists for each _id
        difference_shares_by_id, transpose_list_time_sec = benchmark_time(
            lambda: {
                id: list(map(list, zip(*differences)))
                for id, differences in difference_shares_by_id.items()
            }
        )
        # 3.3 Decrypt and compute distances
        reconstructed, decrypt_time_sec = benchmark_time(
            lambda: [
                {
                    "_id": id,
                    "distances": np.linalg.norm(
                        decrypt_float_list(additive_key, difference_shares)
                    ),
                }
                for id, difference_shares in difference_shares_by_id.items()
            ]
        )
        # 3.4 Sort id list based on the corresponding distances
        sorted_ids = sorted(reconstructed, key=lambda x: x["distances"])

        # Step 4: Query the top num_chunks
        top_num_chunks_ids, top_num_chunks_ids_time_sec = benchmark_time(
            lambda: [item["_id"] for item in sorted_ids[:num_chunks]]
        )
        # 4.1 Query top num_chunks
        chunk_shares, query_top_chunks_time_sec = await benchmark_time_async(
            self.chunk_query_execute, top_num_chunks_ids
        )
        # 4.2 Group chunk shares by ID
        chunk_shares_by_id = group_shares_by_id(
            chunk_shares,  # type: ignore
            lambda share: share["chunk"],
        )
        # 4.3 Decrypt chunks
        top_num_chunks = [
            {"_id": id, "distances": nilql.decrypt(xor_key, chunk_shares)}
            for id, chunk_shares in chunk_shares_by_id.items()
        ]

        # Print benchmarks, if enabled
        if ENABLE_BENCHMARK:
            print(
                f"""Performance breakdown. Time to:\
            \n generate query embedding: {generate_query_embedding_time_sec:.2f} seconds \
            \n check clustering and (if existing) get closest centroid:\
                {cluster_check_and_get_closest_centroid_time_sec:.2f} seconds\
            \n encrypt query embedding: {encrypt_query_embedding_time_sec:.2f} seconds\
            \n ask nilDB to compute the differences: {asking_nildb_time_sec:.2f} seconds\
            \n group shares by Id: {group_shares_by_id_time_sec:.2f} seconds\
            \n transpose list: {transpose_list_time_sec:.2f} seconds\
            \n decrypt: {decrypt_time_sec:.2f} seconds\
            \n get top chunks ids: {top_num_chunks_ids_time_sec:.2f} seconds\
            \n query top chunks: {query_top_chunks_time_sec:.2f} seconds\
            """
            )
        return top_num_chunks

    def nilai_chat_completion(
        self,
        config: ChatCompletionConfig,
    ) -> dict:
        """
        Query the chat completion endpoint of the nilai API.

        Args:
            config (ChatCompletionConfig): Configuration for the chat completion request
            closest_centroid (list[int], optional): The closest centroid vector to filter by

        Returns:
            dict: Chat response from the nilai API
        """
        start_time = time.time()
        # Ensure URL format
        nilai_url = config.nilai_url.rstrip("/") + "/v1/chat/completions"
        # Authorization header
        headers = {
            "Authorization": f"Bearer {config.token}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        # Ensure messages include required roles
        has_system = any(message.get("role") == "system" for message in config.messages)
        has_user = any(message.get("role") == "user" for message in config.messages)
        messages = config.messages.copy()
        if not has_system:
            messages.insert(
                0, {"role": "system", "content": "You are a helpful assistant."}
            )
        if not has_user:
            messages.append({"role": "user", "content": "What is your name?"})
        # Construct the `nilrag` payload
        nilrag = {
            "nodes": [
                {
                    "url": node.url,
                    "bearer_token": node.bearer_token,
                    "schema_id": node.schema_id,
                    "diff_query_id": node.diff_query_id,
                    "clusters_schema_id": node.clusters_schema_id,
                    "cluster_diff_query_id": node.cluster_diff_query_id,
                }
                for node in self.nodes
            ]
        }
        # Construct payload
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": config.stream,
            "nilrag": nilrag,
        }
        try:
            # Send POST request
            print("Sending request to nilai API...")
            request_start = time.time()
            response = requests.post(
                nilai_url, headers=headers, json=payload, timeout=3600
            )
            request_time = time.time() - request_start
            # Handle response
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )
            result = response.json()
            # Debug print response details
            print("Response details:")
            print(f"Status code: {response.status_code}")
            if "usage" in result:
                print(f"Tokens used: {result['usage']['total_tokens']}")
            # Check if we have access to the number of documents processed
            if "nilrag" in result and "documents_processed" in result["nilrag"]:
                print(
                    f"Number of documents processed: {result['nilrag']['documents_processed']}"
                )
            total_time = time.time() - start_time
            print(
                f"Timing breakdown:\
                  \n Total time: {total_time:.2f} seconds\
                  \n Request time: {request_time:.2f} seconds\
                  \n Processing time: {total_time - request_time:.2f} seconds"
            )
            return result
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while querying the chat completion endpoint: {str(e)}"
            ) from e


def euclidean_distance(a: list, b: list):
    """
    Calculate Euclidean distance between two vectors.

    Args:
        a (array-like): First vector
        b (array-like): Second vector

    Returns:
        float: Euclidean distance between vectors a and b
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def find_closest_chunks(
    query_embedding: list, chunks: list, embeddings: list, top_k: int = 2
):
    """
    Find chunks closest to a query embedding using Euclidean distance.

    Args:
        query_embedding (array-like): Embedding vector of the query
        chunks (list): List of text chunks
        embeddings (list): List of embedding vectors for the chunks
        top_k (int, optional): Number of closest chunks to return. Defaults to 2.

    Returns:
        list: List of tuples (chunk, distance) for the top_k closest chunks
    """
    distances = [euclidean_distance(query_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(distances)
    return [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]


def compute_closest_centroid(
    query_embedding: np.ndarray, centroids: list[int]
) -> list[int]:
    """
    Find the closest centroid for a given query embedding.
    Args:
        query_embedding (np.ndarray): The embedding vector of the query in floating-point format
        centroids (list): List of centroid vectors in fixed-point format
    Returns:
        int: The index of the closest centroid
    """
    # Convert query embedding to fixed-point for comparison
    query_embedding_fixed = [to_fixed_point(val) for val in query_embedding]

    # Find closest centroid
    # closest_centroid_idx = None
    closest_centroid_idx = min(
        range(len(centroids)),
        key=lambda i: euclidean_distance(query_embedding_fixed, centroids[i]),
    )

    return centroids[closest_centroid_idx]
