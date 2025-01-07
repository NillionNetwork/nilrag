"""
nilDB class definition for secure data storage and RAG inference.
"""

import base64
import json
import time
from typing import Optional
from uuid import uuid4

import jwt
import requests
from ecdsa import SECP256k1, SigningKey


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
        # pylint: disable=too-many-positional-arguments
        # pylint: disable=too-many-arguments
        self,
        url: str,
        node_id: Optional[str] = None,
        org: Optional[str] = None,
        bearer_token: Optional[str] = None,
        schema_id: Optional[str] = None,
        diff_query_id: Optional[str] = None,
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
        return f"URL: {self.url}\
            \nnode_id: {self.node_id}\
            \norg: {self.org}\
            \nBearer Token: {self.bearer_token}\
            \nSchema ID: {self.schema_id}\
            \nDifferences Query ID: {self.diff_query_id}"


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

    def _init_schema(self):
        """
        Initialize the nilDB schema across all nodes.

        Creates a schema for storing embeddings and chunks with a common schema ID
        across all nilDB nodes. The schema defines the structure for storing document
        embeddings and their corresponding text chunks.

        Raises:
            ValueError: If schema creation fails on any nilDB node
        """
        schema_id = str(
            uuid4()
        )  # the schema_id is assumed to be the same across different nildb instances
        for node in self.nodes:
            node.schema_id = schema_id
            url = node.url + "/schemas"

            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": schema_id,
                "org": node.org,
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
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=3600
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )
            print("Response JSON:", response.json())

    def _init_diff_query(self):
        """
        Initialize the difference query across all nilDB nodes.

        Creates a query that calculates the difference between stored embeddings
        and a query embedding. This query is used for similarity search operations.

        Raises:
            ValueError: If query creation fails on any nilDB node
        """
        diff_query_id = str(
            uuid4()
        )  # the diff_query_id is assumed to be the same across different nildb instances
        for node in self.nodes:
            node.diff_query_id = diff_query_id
            url = node.url + "/queries"

            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            payload = {
                "_id": node.diff_query_id,
                "org": node.org,
                "name": "Returns the difference between the nilDB embeddings and the query embedding",
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
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=3600
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )
            print("Response JSON:", response.json())

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
        for node in self.nodes:
            # Create payload for each node_id
            payload = {
                "iss": node.org,
                "aud": node.node_id,
                "exp": int(time.time()) + ttl,
            }

            # Create and sign the JWT
            node.bearer_token = jwt.encode(payload, signer.to_pem(), algorithm="ES256K")
            print(f"Generated JWT for {node.node_id}: {node.bearer_token}")

    def diff_query_execute(self, nilql_query_embedding: list[list[bytes]]):
        """
        Execute the difference query across all nilDB nodes.

        Args:
            nilql_query_embedding (list): Encrypted query embedding for all nilDB node.
                Example:
                [
                    # First element of embedding vector, shared across nodes
                    [
                        b'share_for_node_0',  # Share for node 0
                        b'share_for_node_1',  # Share for node 1
                        b'share_for_node_2'   # Share for node 2
                    ],
                    # Second element of embedding vector, shared across nodes
                    [
                        b'share_for_node_0',
                        b'share_for_node_1',
                        b'share_for_node_2'
                    ],
                    # And so on for each element in the embedding vector...
                ]

        Returns:
            list: List of difference shares from each nilDB node
                Example:
                [
                    [  # Shares from node 0
                        {
                            '_id': 'f1fc5d71-24a8-4b38-9c5b-0cba1e615acb',
                            'difference': [0, 1, 2, 3, 4]
                        },
                        {
                            '_id': '0997b6f4-ec0c-49fc-8428-1824c496a964',
                            'difference': [5, 6, 7, 8, 9]
                        }
                    ],
                    [  # Shares from node 1
                        {
                            '_id': 'f1fc5d71-24a8-4b38-9c5b-0cba1e615acb',
                            'difference': [10, 11, 12, 13, 14]
                        },
                        {
                            '_id': '0997b6f4-ec0c-49fc-8428-1824c496a964',
                            'difference': [15, 16, 17, 18, 19]
                        }
                    ]
                ]

        Example usage:
            # Initialize secret key for 3 parties
            additive_key = nilql.secret_key({'nodes': [{}] * 3}, {'sum': True})

            # Generate query embedding from text
            query_text = "what is the capital of France?"
            query_embedding = generate_embeddings_huggingface([query_text])[0]

            # Encrypt and share the query embedding
            nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)

            # Execute query and get differences
            difference_shares = nildb.diff_query_execute(nilql_query_embedding)

        Raises:
            ValueError: If query execution fails on any nilDB node
        """

        # Rearrange nilql_query_embedding to group by party
        query_embedding_shares = [
            [entry[party] for entry in nilql_query_embedding]
            for party in range(len(self.nodes))
        ]
        difference_shares = []

        for i, node in enumerate(self.nodes):
            url = node.url + "/queries/execute"
            # Authorization header with the provided token
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }
            diff_query_id = node.diff_query_id

            # Schema payload
            payload = {
                "id": str(diff_query_id),
                "variables": {"query_embedding": query_embedding_shares[i]},
            }

            # Send POST request
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=3600
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )
            try:
                difference_shares_party_i = response.json().get("data")
                if difference_shares_party_i is None:
                    raise ValueError(f"Error in Response: {response.text}")
                difference_shares.append(difference_shares_party_i)
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text

        return difference_shares

    def chunk_query_execute(self, chunk_ids: list[str]):
        """
        Retrieve chunks by their IDs from all nilDB nodes.

        Args:
            chunk_ids (list): List of chunk IDs to retrieve

        Returns:
            list: List of chunk shares from each nilDB node. For example, with 3 nodes and 2 chunks:
                [
                    # Shares from node 1
                    [
                        {   # Same ID across all nodes for the same secret
                            '_id': '123e4567-e89b-12d3-a456-426614174000',
                            'chunk': 'base64EncodedShare1ForChunk1'
                        },
                        {   # Same ID across all nodes for the same secret
                            '_id': '987fcdeb-51a2-43d7-9012-345678901234',
                            'chunk': 'base64EncodedShare1ForChunk2'
                        }
                    ],
                    # Shares from node 2
                    [
                        {
                            '_id': '123e4567-e89b-12d3-a456-426614174000',
                            'chunk': 'base64EncodedShare2ForChunk1'
                        },
                        {
                            '_id': '987fcdeb-51a2-43d7-9012-345678901234',
                            'chunk': 'base64EncodedShare2ForChunk2'
                        }
                    ],
                    # Shares from node 3
                    [
                        {
                            '_id': '123e4567-e89b-12d3-a456-426614174000',
                            'chunk': 'base64EncodedShare3ForChunk1'
                        },
                        {
                            '_id': '987fcdeb-51a2-43d7-9012-345678901234',
                            'chunk': 'base64EncodedShare3ForChunk2'
                        }
                    ]
                ]

        Example:
            # Get top k document IDs from similarity search
            top_k = 2
            top_k_ids = [item['_id'] for item in sorted_ids[:top_k]]

            # Retrieve the chunks for those IDs
            chunk_shares = nilDB.chunk_query_execute(top_k_ids)

        Raises:
            ValueError: If query execution fails on any nilDB node
        """
        chunk_shares = []
        for node in self.nodes:
            url = node.url + "/data/read"
            # Authorization header with the provided token
            headers = {
                "Authorization": "Bearer " + str(node.bearer_token),
                "Content-Type": "application/json",
            }

            # Schema payload
            payload = {"schema": node.schema_id, "filter": {"_id": {"$in": chunk_ids}}}

            # Send POST request
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=3600
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error in POST request: {response.status_code}, {response.text}"
                )
            try:
                chunk_shares_party_i = response.json().get("data")
                if chunk_shares_party_i is None:
                    raise ValueError(f"Error in Response: {response.text}")
                chunk_shares.append(chunk_shares_party_i)
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text

        return chunk_shares

    def upload_data(
        self, lst_embedding_shares: list[list[int]], lst_chunk_shares: list[list[bytes]]
    ):
        """
        Upload embeddings and chunks to all nilDB nodes.

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
            >>> # Upload to nilDB nodes
            >>> nilDB.upload_data(embeddings_shares, chunks_shares)

        Raises:
            AssertionError: If number of embeddings and chunks don't match
            ValueError: If upload fails on any nilDB node
        """
        # Check sizes: same number of embeddings and chunks
        assert len(lst_embedding_shares) == len(
            lst_chunk_shares
        ), f"Mismatch: {len(lst_embedding_shares)} embeddings vs {len(lst_chunk_shares)} chunks."

        for embedding_shares, chunk_shares in zip(
            lst_embedding_shares, lst_chunk_shares
        ):
            # embeddings_shares [384][3]
            # chunks_shares [3][268]

            # 'data_id' has to be the same for every node to allow secret reconstructions
            data_id = str(uuid4())
            for i, node in enumerate(self.nodes):
                url = node.url + "/data/create"
                # Authorization header with the provided token
                headers = {
                    "Authorization": "Bearer " + str(node.bearer_token),
                    "Content-Type": "application/json",
                }
                # Join the shares of one embedding in one vector
                node_i_embedding_shares = [e[i] for e in embedding_shares]
                node_i_chunk_share = chunk_shares[i]
                # encode to be parsed in json
                encoded_node_i_chunk_share = base64.b64encode(
                    node_i_chunk_share
                ).decode("utf-8")
                # Schema payload
                payload = {
                    "schema": node.schema_id,
                    "data": [
                        {
                            "_id": data_id,
                            "embedding": node_i_embedding_shares,
                            "chunk": encoded_node_i_chunk_share,
                        }
                    ],
                }

                # Send POST request
                response = requests.post(
                    url, headers=headers, data=json.dumps(payload), timeout=3600
                )
                if response.status_code != 200:
                    raise ValueError(
                        f"Error in POST request: {response.status_code}, {response.text}"
                    )
                print(
                    {
                        "status_code": response.status_code,
                        "message": "Success",
                        "response_json": response.json(),
                    }
                )

    def nilai_chat_completion(
        # pylint: disable=too-many-positional-arguments
        # pylint: disable=too-many-arguments
        self,
        nilai_url: str,
        token: str,
        messages: list[dict],
        model: str = "Llama-3.2-1B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> dict:
        """
        Query the chat completion endpoint of the nilai API.

        Args:
            nilai_url (str): Base URL for the nilai API
            token (str): Bearer token for authentication
            messages (list[dict]): List of message dictionaries (role and content)
            model (str): AI model to use for completion (default: "Llama-3.2-1B-Instruct")
            temperature (float): Sampling temperature (default: 0.7)
            max_tokens (int): Maximum tokens to generate (default: 2048)
            stream (bool): Whether to stream the response (default: False)

        Returns:
            dict: Chat response from the nilai API
        """
        # Ensure URL format
        nilai_url = nilai_url.rstrip("/") + "/v1/chat/completions"

        # Authorization header
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        # Ensure messages include required roles
        has_system = any(message.get("role") == "system" for message in messages)
        has_user = any(message.get("role") == "user" for message in messages)

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
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
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
