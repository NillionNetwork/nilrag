"""
nilDB class definition for secure data storage and RAG inference.
"""

# pylint: disable=too-many-lines
import asyncio
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
import jwt
import nilql
import numpy as np
import requests
from ecdsa import SECP256k1, SigningKey

from .util import (decrypt_float_list, encrypt_float_list,
                   generate_embeddings_huggingface, get_closest_centroid,
                   group_shares_by_id)

# Benchmark
ENABLE_BENCHMARK = True  # set to False to disable
# Constants
TIMEOUT = 3600
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def benchmark_time(func, *args, **kwargs):
    """Measures the execution time of a sync function."""
    if ENABLE_BENCHMARK:
        start_time = time.time()
    result = func(*args, **kwargs)
    if ENABLE_BENCHMARK:
        return result, time.time() - start_time
    return result, None


async def benchmark_time_async(func, *args, **kwargs):
    """Measures the execution time of an async function."""
    if ENABLE_BENCHMARK:
        start_time = time.time()
    result = await func(*args, **kwargs)
    if ENABLE_BENCHMARK:
        return result, time.time() - start_time
    return result, None


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


# pylint: disable=too-many-instance-attributes
class Node:  # pylint: disable=too-few-public-methods
    """
    Represents a node in the NilDB network.

    A Node contains connection information and identifiers for a specific NilDB instance,
    including the URL endpoint, org identifier, authentication token, and IDs for
    associated schema and queries.

    Attributes:
        url (str): The base URL endpoint for the node, with trailing slash removed
        org (str): The org identifier for this node
        bearer_token (str): Authentication token for API requests
        schema_id (str, optional): ID of the schema associated with this node
        diff_query_id (str, optional): ID of the differences query for this node
        clusters_schema_id (str, optional): ID of the schema of clusters associated with this node
        cluster_diff_query_id (str, optional): ID of the differences query with filter for this node
    """

    def __init__(
        self,
        url: str,
        node_id: Optional[str] = None,
        org: Optional[str] = None,
        bearer_token: Optional[str] = None,
        schema_id: Optional[str] = None,
        diff_query_id: Optional[str] = None,
        clusters_schema_id: Optional[str] = None,
        cluster_diff_query_id: Optional[str] = None,
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
    ):
        """
        Initialize a new Node instance.
        Args:
            url (str): Base URL endpoint for the node
            org (str): org identifier
            bearer_token (str): Authentication token
            schema_id (str, optional): Associated schema ID
            diff_query_id (str, optional): Associated differences query ID
            clusters_schema_id (str, optional): Associated clusters' schema ID
            cluster_diff_query_id (str, optional): Associated differences query with filter ID

        """
        self.url = url[:-1] if url.endswith("/") else url
        self.node_id = node_id
        self.org = org
        self.bearer_token = bearer_token
        self.schema_id = schema_id
        self.diff_query_id = diff_query_id
        self.clusters_schema_id = clusters_schema_id
        self.cluster_diff_query_id = cluster_diff_query_id

    def __repr__(self):
        """
        Returns:
            str: Multi-line string containing all Node attributes
        """
        return f"  URL: {self.url}\
            \n  node_id: {self.node_id}\
            \n  org: {self.org}\
            \n  Bearer Token: {self.bearer_token}\
            \n  Schema ID: {self.schema_id}\
            \n  Differences Query ID: {self.diff_query_id}\
            \n  Clusters' Schema ID: {self.clusters_schema_id}\
            \n  Cluster Differences Query ID: {self.cluster_diff_query_id}\
"


class NilDB:
    """
    A class to manage distributed nilDB nodes for secure data storage and retrieval.

    This class handles initialization, querying, and data upload across multiple nilDB nodes
    while maintaining data security through secret sharing.

    Attributes:
        nodes (list): List of Node instances representing the distributed nilDB nodes
    """

    def __init__(self, nodes: list[Node]):
        """
        Initialize NilDB with a list of nilDB nodes.

        Args:
            nodes (list): List of Node instances representing nilDB nodes
        """
        self.nodes = nodes

    def __repr__(self):
        """Return string representation of NilDB showing all nodes."""
        return "\n".join(
            f"\nNode({i}):\n{repr(node)}" for i, node in enumerate(self.nodes)
        )

    async def init_schema(self):
        """
        Initialize the nilDB schema across all nodes asynchronously.

        Creates a schema for storing cluster centroids (when clustering is performed),
        embeddings and chunks with a common schema ID across all nilDB nodes.
        The schema defines the structure for storing document
        embeddings and their corresponding text chunks and clusters' centroids.

        Raises:
            ValueError: If schema creation fails on any nilDB node
        """
        schema_id = str(uuid4())

        async def create_schema_for_node(node: Node) -> None:
            url = node.url + "/schemas"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": schema_id,
                "name": "nilrag data",
                "keys": ["_id"],
                "schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "NILLION USERS",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "_id": {"type": "string", "format": "uuid", "coerce": True},
                            "cluster_centroid": {
                                "description": "Embedding of the clusters' centroid",
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "embedding": {
                                "description": "Chunks embeddings",
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "chunk": {
                                "type": "string",
                                "description": "Chunks of text inserted by the user",
                            },
                        },
                        "required": ["_id", "embedding", "chunk"],
                        "additionalProperties": False,
                    },
                },
            }
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status not in [
                                HTTPStatus.CREATED,
                                HTTPStatus.OK,
                            ]:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            node.schema_id = schema_id
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to create schema after {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Create schema on all nodes in parallel
        tasks = [create_schema_for_node(node) for node in self.nodes]
        await asyncio.gather(*tasks)
        print(f"Schema {schema_id} created successfully.")
        return schema_id

    async def init_clusters_schema(self):
        """
        Initialize the clusters' schema across all nodes asynchronously.

        Creates a clusters' schema for storing clusters' centroids embeddings
        associated to a common schema ID.
        The cluster schema defines the structure for storing the
        embeddings of cluster centroids.

        Raises:
            ValueError: If clusters' schema creation fails on any nilDB node
        """
        clusters_schema_id = str(uuid4())

        async def create_clusters_schema_for_node(node: Node) -> None:
            url = node.url + "/schemas"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": clusters_schema_id,
                "name": "Clusters' centroids",
                "schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "CLUSTERS CENTROIDS",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "_id": {"type": "string", "format": "uuid", "coerce": True},
                            "cluster_centroid": {
                                "description": "Embedding of the clusters centroid",
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["_id", "cluster_centroid"],
                        "additionalProperties": False,
                    },
                },
            }
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status not in [
                                HTTPStatus.CREATED,
                                HTTPStatus.OK,
                            ]:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            node.clusters_schema_id = clusters_schema_id
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to create clusters' schema after\
                                {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Create clusters' schema on all nodes in parallel
        tasks = [create_clusters_schema_for_node(node) for node in self.nodes]
        await asyncio.gather(*tasks)
        print(f"Clusters' Schema {clusters_schema_id} created successfully.")
        return clusters_schema_id

    async def init_diff_query(self):
        """
        Initialize the difference query across all nilDB nodes asynchronously.

        Creates a query that calculates the difference between stored embeddings
        and a query embedding. This query is used for similarity search operations.

        Raises:
            ValueError: If query creation fails on any nilDB node
        """
        diff_query_id = str(uuid4())

        async def create_query_for_node(node: Node) -> None:
            url = node.url + "/queries"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": diff_query_id,
                "name": (
                    "Returns the difference between the nilDB embeddings "
                    "and the query embedding"
                ),
                "schema": node.schema_id,
                "variables": {
                    "query_embedding": {
                        "description": "The query embedding",
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "pipeline": [
                    {"$addFields": {"query_embedding": "##query_embedding"}},
                    {
                        "$project": {
                            "_id": 1,
                            "difference": {
                                "$map": {
                                    "input": {
                                        "$zip": {
                                            "inputs": ["$embedding", "$query_embedding"]
                                        }
                                    },
                                    "as": "pair",
                                    "in": {
                                        "$subtract": [
                                            {"$arrayElemAt": ["$$pair", 0]},
                                            {"$arrayElemAt": ["$$pair", 1]},
                                        ]
                                    },
                                }
                            },
                        }
                    },
                ],
            }
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status not in [
                                HTTPStatus.CREATED,
                                HTTPStatus.OK,
                            ]:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            node.diff_query_id = diff_query_id
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to create query after {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Create query on all nodes in parallel
        tasks = [create_query_for_node(node) for node in self.nodes]
        await asyncio.gather(*tasks)
        print(f"Query {diff_query_id} created successfully.")
        return diff_query_id

    async def init_cluster_diff_query(self):
        """
        Initialize the difference query across all nilDB nodes asynchronously.
        Creates a query that calculates the difference between stored embeddings
        with closest centroid filter and a query embedding.
        This query is used for similarity search operations.

        Raises:
            ValueError: If query creation fails on any nilDB node
        """
        cluster_diff_query_id = str(uuid4())

        async def create_cluster_query_for_node(node: Node) -> None:
            url = node.url + "/queries"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": cluster_diff_query_id,
                "name": (
                    "Returns the difference between the nilDB embeddings "
                    "and the query embedding with a closest centroid tag"
                ),
                "schema": node.schema_id,
                "variables": {
                    "query_embedding": {
                        "description": "The query embedding",
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "closest_centroid": {
                        "description": "The closest centroid to match",
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "pipeline": [
                    {"$addFields": {"query_embedding": "##query_embedding"}},
                    {
                        "$match": {
                            "$expr": {
                                "$eq": ["$cluster_centroid", "##closest_centroid"]
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "difference": {
                                "$map": {
                                    "input": {
                                        "$zip": {
                                            "inputs": ["$embedding", "$query_embedding"]
                                        }
                                    },
                                    "as": "pair",
                                    "in": {
                                        "$subtract": [
                                            {"$arrayElemAt": ["$$pair", 0]},
                                            {"$arrayElemAt": ["$$pair", 1]},
                                        ]
                                    },
                                }
                            },
                        }
                    },
                ],
            }
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status not in [
                                HTTPStatus.CREATED,
                                HTTPStatus.OK,
                            ]:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            node.cluster_diff_query_id = cluster_diff_query_id
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to create query after {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Create query with filter on all nodes in parallel
        tasks = [create_cluster_query_for_node(node) for node in self.nodes]
        await asyncio.gather(*tasks)
        print(f"Cluster query {cluster_diff_query_id} created successfully.")
        return cluster_diff_query_id

    def generate_jwt(self, secret_key: str, ttl: int = 3600):
        """
        Create JWTs signed with ES256K for multiple node_ids.

        Args:
            secret_key: Secret key in hex format
            org_did: Issuer's DID
            node_ids: List of node IDs (audience)
            ttl: Time-to-live for the JWT in seconds
        """
        # Convert the secret key from hex to bytes
        private_key = bytes.fromhex(secret_key)
        signer = SigningKey.from_string(private_key, curve=SECP256k1)
        jwts = []
        for node in self.nodes:
            # Create payload for each node_id
            payload = {
                "iss": node.org,
                "aud": node.node_id,
                "exp": int(time.time()) + ttl,
            }

            # Create and sign the JWT
            node.bearer_token = jwt.encode(payload, signer.to_pem(), algorithm="ES256K")
            jwts.append(node.bearer_token)
        return jwts

    async def diff_query_execute(
        self,
        nilql_query_embedding: list[list[bytes]],
        closest_centroid: list[int] = None,
    ) -> List:
        """
        Execute the difference query across all nilDB nodes asynchronously.

        Args:
            nilql_query_embedding (list): Encrypted query embedding for all nilDB node.
            closest_centroid (list): The closest centroid to filter by

        Returns:
            list: List of difference shares from each nilDB node.

        Raises:
            ValueError: If query execution fails on any nilDB node
        """
        # Rearrange nilql_query_embedding to group by party
        query_embedding_shares = [
            [entry[party] for entry in nilql_query_embedding]
            for party in range(len(self.nodes))
        ]

        async def execute_query_on_node(
            node: Node, node_index: int
        ) -> List[Dict[str, Any]]:
            url = node.url + "/queries/execute"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            # Assemble the variables to execute query
            variables = {"query_embedding": query_embedding_shares[node_index]}
            # If there was clustering adds variable "closest_centroid" and chooses
            # cluster_diff_query_id to perform the query. otherwise, chooses diff_query_id
            if closest_centroid is not None:
                # adds "closesr_centroid"
                variables["closest_centroid"] = closest_centroid
                if not node.cluster_diff_query_id:
                    raise ValueError(f"[Node {node.url}] Missing cluster_diff_query_id")
                query_id = str(node.cluster_diff_query_id)
            else:
                if not node.diff_query_id:
                    raise ValueError(f"[Node {node.url}] Missing diff_query_id")
                query_id = str(node.diff_query_id)
            payload = {"id": query_id, "variables": variables}
            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status != HTTPStatus.OK:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            result = await response.json()
                            if result.get("data") is None:
                                raise ValueError(f"Error in Response: {result}")
                            return result.get("data", [])
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to execute query after {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Execute queries on all nodes in parallel
        tasks = [execute_query_on_node(node, i) for i, node in enumerate(self.nodes)]
        difference_shares = await asyncio.gather(*tasks)
        return difference_shares

    async def chunk_query_execute(self, chunk_ids: list[str]) -> List:
        """
        Retrieve chunks by their IDs from all nilDB nodes asynchronously.

        Args:
            chunk_ids (list): List of chunk IDs to retrieve

        Returns:
            list: List of chunk shares from each nilDB node.

        Raises:
            ValueError: If query execution fails on any nilDB node
        """

        async def read_from_node(node: Node) -> List[Dict[str, Any]]:
            url = node.url + "/data/read"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {"schema": node.schema_id, "filter": {"_id": {"$in": chunk_ids}}}

            for attempt in range(MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload, timeout=TIMEOUT
                        ) as response:
                            if response.status != HTTPStatus.OK:
                                error_text = await response.text()
                                raise ValueError(
                                    f"Error in POST request: {response.status}, {error_text}"
                                )
                            result = await response.json()
                            if result.get("data") is None:
                                raise ValueError(f"Error in Response: {result}")
                            return result.get("data", [])
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:
                        raise ValueError(
                            f"Failed to read chunks after {MAX_RETRIES} attempts: {str(e)}"
                        ) from e
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Read from all nodes in parallel
        tasks = [read_from_node(node) for node in self.nodes]
        chunk_shares = await asyncio.gather(*tasks)
        return chunk_shares

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    async def upload_data(
        self,
        lst_embedding_shares: list[list[int]],
        lst_chunk_shares: list[list[bytes]],
        batch_size: int = 100,
        labels: list[int] | None = None,
        centroids: list[int] | None = None,
    ) -> None:
        """
        Upload embeddings and chunks to all nilDB nodes asynchronously in batches.

        Args:
            lst_embedding_shares (list): List of embedding shares for each document,
                e.g. for 3 nodes:
                [
                    # First document's embedding vector (384 dimensions)
                    [
                        # First dimension split into 3 shares (sum mod 2^32)
                        [1234567890, 987654321, 2072745085],
                        # Second dimension split into 3 shares (sum mod 2^32)
                        [3141592653, 2718281828, 3435092815],
                        # ... 382 more dimensions, each split into 3 shares
                    ],
                    # More documents...
                ]
            lst_chunk_shares (list): List of chunk shares for each document, e.g. for 3 nodes:
                [
                    [  # First document's chunk shares
                        b"encrypted chunk for node 1",
                        b"encrypted chunk for node 2",
                        b"encrypted chunk for node 3"
                    ],
                    # More documents...
                ]
            batch_size (int, optional): Number of documents to upload in each batch.
                Defaults to 100.
            labels (list[int]): List of labels given for each embedding.
                Defaults to None.
            centroids (list[int]): List of clusters centroids when clustering is performed.
                Defaults to None.

        Example:
            >>> # Set up encryption keys for 3 nodes
            >>> additive_key = nilql.secret_key({'nodes': [{}] * 3}, {'sum': True})
            >>> xor_key = nilql.secret_key({'nodes': [{}] * 3}, {'store': True})
            >>>
            >>> # Generate embeddings and chunks
            >>> chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)
            >>> # Each embedding is 384-dimensional
            >>> embeddings = generate_embeddings_huggingface(chunks)
            >>>
            >>> # Create shares
            >>> chunks_shares = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
            >>> embeddings_shares = [encrypt_float_list(additive_key, emb) for emb in embeddings]
            >>>
            >>> # Upload to nilDB nodes in batches of 100
            >>> await nilDB.upload_data(embeddings_shares, chunks_shares, batch_size=100)

        Raises:
            AssertionError: If number of embeddings and chunks don't match
            ValueError: If upload fails on any nilDB node
        """
        await self.check_inputs_to_upload(
            lst_embedding_shares, lst_chunk_shares, labels, centroids
        )

        # pylint: disable=too-many-locals
        async def process_batch(batch_start: int, batch_end: int) -> None:
            """Process and upload a single batch of documents."""
            print(
                f"Processing batch {batch_start//batch_size + 1}: "
                f"documents {batch_start} to {batch_end}"
            )
            # Generate document IDs for this batch
            doc_ids = [str(uuid4()) for _ in range(batch_start, batch_end)]
            tasks = []
            for node_idx, node in enumerate(self.nodes):
                batch_data = []
                for batch_idx, doc_idx in enumerate(range(batch_start, batch_end)):
                    # Join the shares of one embedding in one vector for this node
                    batch_entry = {
                        "_id": doc_ids[batch_idx],
                        "embedding": [
                            e[node_idx] for e in lst_embedding_shares[doc_idx]
                        ],
                        "chunk": lst_chunk_shares[doc_idx][node_idx],
                    }
                    # In case clustering is performed,
                    # join the clusters centroid of the corresponding embedding
                    if (
                        labels is not None
                        and centroids is not None
                        and len(centroids) > 1
                    ):
                        batch_entry["cluster_centroid"] = centroids[labels[doc_idx]]
                    # Add this entry to the batch data
                    batch_data.append(batch_entry)
                tasks.append(self.upload_to_node(node, batch_data))
            try:
                results = await asyncio.gather(*tasks)
                print(f"Successfully uploaded batch {batch_start//batch_size + 1}")
                for result in results:
                    print(
                        {
                            "status_code": 200,
                            "message": "Success",
                            "response_json": result,
                        }
                    )
            except Exception as e:
                print(f"Error uploading batch {batch_start//batch_size + 1}: {str(e)}")
                raise

        # Process data in batches
        total_documents = len(lst_embedding_shares)
        for batch_start in range(0, total_documents, batch_size):
            batch_end = min(batch_start + batch_size, total_documents)
            await process_batch(batch_start, batch_end)
        # After processing all batches, upload centroids if they exist
        if centroids is not None and len(centroids) > 1:
            await self.upload_all_centroids(centroids)

    async def get_closest_centroid(
        self, query_embedding: np.ndarray
    ) -> tuple[int, Optional[List[float]]]:
        """
        Check if clustering was performed and return the number of clusters.

        Returns:
            int: Number of clusters found (0 if no clustering was performed)
        """

        async def read_clusters_from_node(node: Node) -> dict:
            # Checking clusters in node {node.url}
            # Using clusters schema ID: {node.clusters_schema_id}

            url = node.url + "/data/read"
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {"schema": node.clusters_schema_id, "filter": {}}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(
                            f"Error reading clusters: {response.status}, {error_text}"
                        )
                    result = await response.json()
                    return result

        try:
            # Read from all nodes
            tasks = [read_clusters_from_node(node) for node in self.nodes]
            results = await asyncio.gather(*tasks)
            # Get centroids from the first node's result
            clusters_data = results[0].get("data", [])
            if not clusters_data:
                # No clusters found
                return 0, None
            num_clusters = len(clusters_data)
            centroids = [centroid["cluster_centroid"] for centroid in clusters_data]
            closest_centroid = get_closest_centroid(query_embedding, centroids)
            return num_clusters, closest_centroid
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
            print(f"Error checking clusters and finding closest centroid: {str(e)}")
            return 0, None

    async def check_inputs_to_upload(
        self,
        embedding_shares: list,
        chunk_shares: list,
        labels: list | None,
        centroids: list | None,
    ) -> None:
        """
        Checks if the number of embeddings and chunks is the same
        when there is clustering, check if each embedding was assigned a label.

        """
        # Check sizes: same number of embeddings and chunks
        assert len(embedding_shares) == len(
            chunk_shares
        ), f"Mismatch: {len(embedding_shares)} embeddings vs {len(chunk_shares)} chunks."
        # Check that the number of labels of each cluster matches the number of embeddings,
        # if labels are provided
        if labels is not None:
            assert len(labels) == len(
                embedding_shares
            ), f"Mismatch: {len(labels)} labels vs {len(embedding_shares)} embeddings."
        # Check that the number of centroids is correct if provided
        if centroids is not None:
            assert (
                len(centroids) > 1
            ), "Centroids must be provided when clustering is enabled."

    async def upload_to_node(self, node: Node, batch_data: list[dict]):
        """Upload a batch of data to a specific node."""
        url = node.url + "/data/create"
        headers = {
            "Authorization": "Bearer " + str(node.bearer_token),
            "Content-Type": "application/json",
        }

        payload = {
            "schema": node.schema_id,
            "data": batch_data,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Error in POST request: {response.status}, {error_text}"
                    )
                return await response.json()

    async def upload_centroids_to_node(self, node: Node, centroids_data: list[int]):
        """Upload centroids to the clusters schema."""
        url = node.url + "/data/create"
        headers = {
            "Authorization": "Bearer " + str(node.bearer_token),
            "Content-Type": "application/json",
        }
        payload = {
            "schema": node.clusters_schema_id,
            "data": centroids_data,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Error in POST request: {response.status}, {error_text}"
                    )
                return await response.json()

    async def upload_all_centroids(self, centroids: list[int]) -> None:
        """
        Uploads the centroids to all nodes after generating the appropriate IDs.
        """
        print("Uploading centroids to Clusters' Schema...")

        # Generate IDs for centroids
        centroid_ids = [str(uuid4()) for _ in centroids]

        # Collect tasks for uploading centroids
        tasks = []
        for _, node in enumerate(self.nodes):
            centroids_data = [
                {"_id": centroid_ids[centroid_idx], "cluster_centroid": centroid}
                for centroid_idx, centroid in enumerate(centroids)
            ]
            tasks.append(self.upload_centroids_to_node(node, centroids_data))

        # Gather the results of all upload tasks
        try:
            results = await asyncio.gather(*tasks)
            print("Successfully uploaded centroids")
            for result in results:
                print(
                    {
                        "status_code": 200,
                        "message": "Success",
                        "response_json": result,
                    }
                )
        except Exception as e:
            print(f"Error uploading centroids: {str(e)}")
            raise

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
