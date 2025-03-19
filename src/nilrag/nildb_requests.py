"""
nilDB class definition for secure data storage and RAG inference.
"""

import asyncio
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
import jwt
import requests
from ecdsa import SECP256k1, SigningKey

# Constants
TIMEOUT = 3600
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


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
    """

    def __init__(
        self,
        url: str,
        node_id: Optional[str] = None,
        org: Optional[str] = None,
        bearer_token: Optional[str] = None,
        schema_id: Optional[str] = None,
        diff_query_id: Optional[str] = None,
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
        """
        self.url = url[:-1] if url.endswith("/") else url
        self.node_id = node_id
        self.org = org
        self.bearer_token = bearer_token
        self.schema_id = schema_id
        self.diff_query_id = diff_query_id

    def __repr__(self):
        """
        Returns a string representation of the Node instance.

        Returns:
            str: Multi-line string containing all Node attributes
        """
        return f"  URL: {self.url}\
            \n  node_id: {self.node_id}\
            \n  org: {self.org}\
            \n  Bearer Token: {self.bearer_token}\
            \n  Schema ID: {self.schema_id}\
            \n  Differences Query ID: {self.diff_query_id}"


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

        Creates a schema for storing embeddings and chunks with a common schema ID
        across all nilDB nodes. The schema defines the structure for storing document
        embeddings and their corresponding text chunks.

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
                            if response.status not in [HTTPStatus.CREATED, HTTPStatus.OK]:
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
                    }
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
                            if response.status not in [HTTPStatus.CREATED, HTTPStatus.OK]:
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

    async def diff_query_execute(self, nilql_query_embedding: list[list[bytes]]):
        """
        Execute the difference query across all nilDB nodes asynchronously.

        Args:
            nilql_query_embedding (list): Encrypted query embedding for all nilDB node.

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
            payload = {
                "id": str(node.diff_query_id),
                "variables": {"query_embedding": query_embedding_shares[node_index]},
            }

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

    async def chunk_query_execute(self, chunk_ids: list[str]):
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

    async def upload_data(
        self,
        lst_embedding_shares: list[list[int]],
        lst_chunk_shares: list[list[bytes]],
        batch_size: int = 100
    ):
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
        # Check sizes: same number of embeddings and chunks
        assert len(lst_embedding_shares) == len(
            lst_chunk_shares
        ), f"Mismatch: {len(lst_embedding_shares)} embeddings vs {len(lst_chunk_shares)} chunks."

        async def upload_to_node(node: Node, batch_data: list[dict]):
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
                    # Use the pre-generated ID for this document
                    data_id = doc_ids[batch_idx]
                    # Join the shares of one embedding in one vector for this node
                    node_embedding_shares = [
                        e[node_idx] for e in lst_embedding_shares[doc_idx]
                    ]
                    encoded_node_chunk_share = lst_chunk_shares[doc_idx][node_idx]

                    batch_data.append({
                        "_id": data_id,
                        "embedding": node_embedding_shares,
                        "chunk": encoded_node_chunk_share,
                    })

                task = upload_to_node(node, batch_data)
                tasks.append(task)

            try:
                results = await asyncio.gather(*tasks)
                print(f"Successfully uploaded batch {batch_start//batch_size + 1}")
                for result in results:
                    print({
                        "status_code": 200,
                        "message": "Success",
                        "response_json": result
                    })
            except Exception as e:
                print(f"Error uploading batch {batch_start//batch_size + 1}: {str(e)}")
                raise

        # Process data in batches
        total_documents = len(lst_embedding_shares)
        for batch_start in range(0, total_documents, batch_size):
            batch_end = min(batch_start + batch_size, total_documents)
            await process_batch(batch_start, batch_end)

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
            response = requests.post(
                nilai_url, headers=headers, json=payload, timeout=3600
            )

            # Handle response
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )

            return response.json()  # Return the parsed JSON response
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while querying the chat completion endpoint: {str(e)}"
            ) from e
