"""
This module provides the RAGVault class, which is a wrapper around the SecretVaultWrapper,
NilDBInit, and NilDBOps classes.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import nilql
import numpy as np
import requests
from dotenv import set_key
from secretvaults import SecretVaultWrapper

from .nildb.initialization import NilDBInit
from .nildb.operations import NilDBOps
from .utils.benchmark import benchmark_time, benchmark_time_async
from .utils.process import (create_chunks, generate_embeddings_huggingface,
                            load_file)
from .utils.transform import (decrypt_float_list, encrypt_float_list,
                              group_shares_by_id, to_fixed_point)

# Constants
TIMEOUT = 3600


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
    """
    RAGVault is a wrapper around the SecretVaultWrapper, NilDBInit, and NilDBOps classes.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        nodes: List[Dict[str, str]],
        credentials: Dict[str, str],
        *args,
        with_clustering: bool | None = None,
        clusters_schema_id: str | None = None,
        subtract_query_id: str | None = None,
        **kwargs,
    ):
        # SecretVaultWrapper args first
        super().__init__(nodes, credentials, *args, **kwargs)
        # Handle your RAGVault flags
        self.clusters_schema_id = clusters_schema_id
        self.with_clustering = with_clustering
        self.subtract_query_id = subtract_query_id

    # pylint: disable=too-many-arguments
    @classmethod
    async def create(
        cls,
        nodes: list[dict],
        credentials: dict,
        *args,
        with_clustering: bool | None = None,
        clusters_schema_id: str | None = None,
        subtract_query_id: str | None = None,
        **kwargs,
    ) -> "RAGVault":
        """
        Create a RAGVault instance.
        """
        # Construct object synchronously
        self = cls(
            nodes,
            credentials,
            *args,
            with_clustering=with_clustering,
            clusters_schema_id=clusters_schema_id,
            subtract_query_id=subtract_query_id,
            **kwargs,
        )
        # Perform async initialization from SecretVaultWrapper (await SecretVaultWrapper.init())
        await self.init()
        return self

    @classmethod
    async def bootstrap(
        cls,
        org_config: Dict[str, str],
        with_clustering: bool | None = None,
        env_path: str = ".env",
    ) -> "RAGVault":
        """
        First-time setup that allows to:
          • Instantiate & initialize the RAG vault with clustering
          • Create RAG schema, clusters schema, and subtract query
          • Write all required keys into .env in one pass

        Returns:
          (rag_vault, schema_id, clusters_schema_id, subtract_query_id)
        """
        # Ensure .env exists
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"{env_path} not found")

        # Instantiate & await‑init the vault
        rag = await cls.create(
            org_config["nodes"],
            org_config["org_credentials"],
            with_clustering=with_clustering,
        )

        # Create schemas & query
        schema_id = await rag.create_rag_schema()
        subtract_query_id = await rag.create_subtract_query()
        clusters_schema_id = (
            await rag.create_clusters_schema() if with_clustering else None
        )

        updates = {
            "SCHEMA_ID": schema_id,
            "CLUSTERS_SCHEMA_ID": clusters_schema_id if with_clustering else "",
            "QUERY_ID": subtract_query_id,
        }

        for key, value in updates.items():
            set_key(env_path, key, value)

        return rag, schema_id, clusters_schema_id, subtract_query_id

    async def process_rag_data(
        self,
        file_path: str,
        chunk_size: int = 50,
        overlap: int = 10,
    ) -> Tuple[List[List[float]], List[List[bytes]], List[List[bytes]]]:
        """
        Process RAG data.

        This function:
        1. Loads the input file
        2. Creates chunks from the input file
        3. Generates embeddings for the chunks
        4. Encrypts the chunks and embeddings
        """
        # Initialize secret keys for different modes of operation
        num_nodes = len(self.nodes)
        additive_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_nodes}, {"sum": True}
        )
        xor_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_nodes}, {"store": True}
        )

        # Load and process the input file
        paragraphs = load_file(file_path)
        chunks = create_chunks(paragraphs, chunk_size=chunk_size, overlap=overlap)

        # Generate embeddings and chunks
        embeddings = generate_embeddings_huggingface(chunks)

        # Encrypt chunks and embeddings
        chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
        embeddings_shares = [
            encrypt_float_list(additive_key, embedding) for embedding in embeddings
        ]

        return embeddings, embeddings_shares, chunks_shares

    async def get_closest_centroids(
        self, query_embedding: np.ndarray, num_closest_centroids: int = 1
    ) -> Tuple[int, Optional[List[int]]]:
        """
        Check if clustering was performed and return the number of clusters.

        Args:
            query_embedding (np.ndarray): Embedding vector of the query
            num_closest_centroids (int, optional): Number of closest centroids to return. Defaults to 1.

        Returns:
            int: Number of clusters found (0 if no clustering was performed)
            list[int]: List of closest centroids (None if no clustering was performed)
        """

        try:
            # Check if clustering was performed
            if not self.clusters_schema_id:
                return 0, None
            # Read from the first node
            node = self.nodes[0]
            schema_id = self.clusters_schema_id
            data_filter = {}
            clusters_data = await self.read_from_node(node, schema_id, data_filter)
            if not clusters_data:
                # No clusters found
                return 0, None
            centroids = [centroid["cluster_centroid"] for centroid in clusters_data]
            closest_centroids = compute_closest_centroids(
                query_embedding, centroids, num_closest_centroids
            )
            return len(clusters_data), closest_centroids
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
            print(f"Error checking clusters and finding closest centroid: {str(e)}")
            return 0, None

    # pylint: disable=too-many-locals
    async def top_num_chunks_execute(
        self,
        query: str,
        num_chunks: int,
        enable_benchmark: bool = False,
        num_clusters: int = 1,
    ) -> List:
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
            enable_benchmark (bool, optional): Whether to benchmark the process. Defaults to False.
            num_clusters (int, optional): The number of clusters to consider. Defaults to 1.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - `_id` (Any): The unique identifier of the data chunk.
                - `distances` (float): The computed distance between the query and the data chunk.
        """
        # Check the input format
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
            generate_embeddings_huggingface, query, enable=enable_benchmark
        )

        # 1.2 Check if clustering was performed and (if existing) get the closest centroids
        (
            num_clusters,
            closest_centroids,
        ), cluster_check_and_get_closest_centroid_time_sec = await benchmark_time_async(
            self.get_closest_centroids,
            query_embedding,
            num_clusters,
            enable=enable_benchmark,
        )

        # 1.3 Encrypt query embedding
        nilql_query_embedding, encrypt_query_embedding_time_sec = benchmark_time(
            encrypt_float_list, additive_key, query_embedding, enable=enable_benchmark
        )

        # Step 2: Ask NilDB to compute the differences
        difference_shares, asking_nildb_time_sec = await benchmark_time_async(
            self.execute_subtract_query,
            nilql_query_embedding,
            closest_centroids,
            enable=enable_benchmark,
        )

        # Step 3: Compute distances and sort
        # 3.1 Group difference shares by ID
        difference_shares_by_id, group_shares_by_id_time_sec = benchmark_time(
            group_shares_by_id,
            difference_shares,
            lambda share: share["difference"],
            enable=enable_benchmark,
        )

        # 3.2 Transpose the lists for each _id
        difference_shares_by_id, transpose_list_time_sec = benchmark_time(
            lambda: {
                id: list(map(list, zip(*differences)))
                for id, differences in difference_shares_by_id.items()
            },
            enable=enable_benchmark,
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
            ],
            enable=enable_benchmark,
        )
        # 3.4 Sort id list based on the corresponding distances
        sorted_ids = sorted(reconstructed, key=lambda x: x["distances"])

        # Step 4: Query the top num_chunks
        top_num_chunks_ids, top_num_chunks_ids_time_sec = benchmark_time(
            lambda: [item["_id"] for item in sorted_ids[:num_chunks]],
            enable=enable_benchmark,
        )

        # 4.1 Query top num_chunks
        chunk_shares, query_top_chunks_time_sec = await benchmark_time_async(
            self.read_chunk_from_nodes, top_num_chunks_ids, enable=enable_benchmark
        )
        # 4.2 Group chunk shares by ID
        chunk_shares_by_id = group_shares_by_id(
            chunk_shares,  # type: ignore
            lambda share: share["chunk"],
        )
        # 4.3 Decrypt chunks
        top_num_chunks = [
            {"_id": id, "chunks": nilql.decrypt(xor_key, chunk_shares)}
            for id, chunk_shares in chunk_shares_by_id.items()
        ]

        # Print benchmarks, if enabled
        if enable_benchmark:
            print(
                f"""Performance breakdown. Time to:\
            \n generate query embedding: {generate_query_embedding_time_sec:.2f} seconds \
            \n check clustering and (if existing) get closest centroid:\
                {cluster_check_and_get_closest_centroid_time_sec:.2f} seconds\
            \n Number of clusters found: {num_clusters}\
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
                    "url": node["url"],
                    "did": node["did"],
                }
                for node in self.nodes
            ],
            "org_secret_key": self.credentials["secret_key"],
            "org_did": self.credentials["org_did"],
            "schema_id": self.schema_id,
            "clusters_schema_id": self.clusters_schema_id,
            "subtract_query_id": self.subtract_query_id,
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
                nilai_url, headers=headers, json=payload, timeout=TIMEOUT
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


def compute_closest_centroids(
    query_embedding: np.ndarray, centroids: List[int], num_closest_centroids: int = 1
) -> List[int]:
    """
    Find the k closest centroids for a given query embedding.

    Args:
        query_embedding (np.ndarray): The embedding vector of the query in floating-point format
        centroids (list): List of centroid vectors in fixed-point format
        num_closest_centroids (int, optional): Number of closest centroids to return. Defaults to 1.

    Returns:
        list[int]: The indices of the closest centroids
    """
    # Convert query embedding to fixed-point for comparison
    query_embedding_fixed = [to_fixed_point(val) for val in query_embedding]

    # Find closest centroids
    closest = sorted(
        range(len(centroids)),
        key=lambda i: euclidean_distance(query_embedding_fixed, centroids[i]),
    )

    # Take top closest
    return closest[:num_closest_centroids]
