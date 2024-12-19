import requests
import json
from uuid import uuid4
import base64
import os
import nilql
from .util import create_chunks, decrypt_float_list, decrypt_string_list, encrypt_float_list, encrypt_string_list, generate_embeddings_huggingface, load_file, to_fixed_point

class Node:
    def __init__(self, url, owner, bearer_token, schema_id=None, diff_query_id=None):
        self.url = url[:-1] if url.endswith('/') else url
        self.owner = str(owner)
        self.bearer_token = str(bearer_token)
        self.schema_id = schema_id
        self.diff_query_id = diff_query_id


    def __repr__(self):
        return f"URL: {self.url}\
            \nowner: {self.owner}\
            \nBearer Token: {self.bearer_token}\
            \nSchema ID: {self.schema_id}\
            \nDifferences Query ID: {self.diff_query_id}"

    def to_dict(self):
        """Convert Node to a dictionary for serialization."""
        return {
            "url": self.url,
            "owner": self.owner,
            "bearer_token": self.bearer_token,
            "schema_id": self.schema_id,
            "diff_query_id": self.diff_query_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Node instance from a dictionary."""
        node = cls(data["url"], data["owner"], data["bearer_token"])
        node.schema_id = data.get("schema_id")
        node.diff_query_id = data.get("diff_query_id")
        return node

class NilDB:
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return "\n".join(f"\nNode({i}):\n{repr(node)}" for i, node in enumerate(self.nodes))

    def to_json(self, file_path):
        """Serialize NilDB to JSON and save to a file."""
        data = {
            "nodes": [node.to_dict() for node in self.nodes]
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, file_path):
        """Deserialize NilDB from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        nodes = [Node.from_dict(node_data) for node_data in data["nodes"]]
        return cls(nodes)

    def init_schema(self):
        # Send POST request
        schema_id = str(uuid4()) # the schema_id is assumed to be the same accross different nildb instances
        node.schema_id = schema_id
        for node in self.nodes:
            url = node.url + "/schemas"

            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }
            payload = {
                "_id": schema_id,
                "owner": node.owner,
                "name": "nilrag data",
                "keys": ["_id"],
                "schema": {
                    "$schema": "http://json-schema.owner/draft-07/schema#",
                    "title": "NILLION USERS",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "_id": {
                                "type": "string",
                                "format": "uuid",
                                "coerce": True
                            },
                            "embedding": {
                                "description": "Chunks embeddings",
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "chunk": {
                                "type": "string",
                                "description": "Chunks of text inserted by the user"
                            }
                        },
                        "required": ["_id", "embedding", "chunk"],
                        "additionalProperties": False
                    }
                }
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ValueError(f"Error in POST request: {response.status_code}, {response.text}")
            print("Response JSON:", response.json())


    def init_diff_query(self):
        # Send POST request
        diff_query_id = str(uuid4()) # the diff_query_id is assumed to be the same accross different nildb instances
        node.diff_query_id = diff_query_id
        for node in self.nodes:
            url = node.url + "/queries"

            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }
            payload = {
                "_id": node.diff_query_id,
                "owner": node.owner,
                "name": "Returns the difference between the nilDB embeddings and the query embedding",
                "schema": node.schema_id,
                "variables": {
                    "query_embedding": {
                        "description": "The query embedding",
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    }
                },
                "pipeline": [
                    {
                        "$addFields": {
                            "query_embedding": "##query_embedding"
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "difference": {
                                "$map": {
                                    "input": {
                                        "$zip": {
                                        "inputs": [
                                            "$embedding",
                                            "$query_embedding"
                                        ]
                                        }
                                    },
                                    "as": "pair",
                                    "in": {
                                        "$subtract": [
                                            {
                                                "$arrayElemAt": ["$$pair", 0]
                                            },
                                            {
                                                "$arrayElemAt": ["$$pair", 1]
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                ]
            }
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ValueError(f"Error in POST request: {response.status_code}, {response.text}")
            print("Response JSON:", response.json())


    def diff_query_execute(self, query_embedding_shares):
        
        difference_shares = []

        for i, node in enumerate(self.nodes):
            url = node.url + "/queries/execute"
            # Authorization header with the provided token
            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }
            diff_query_id = node.diff_query_id

            # Schema payload
            payload = {
                    "id": str(diff_query_id),
                    "variables": {
                        "query_embedding": query_embedding_shares[i]
                    }
            }

            # Send POST request
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ValueError(f"Error in POST request: {response.status_code}, {response.text}")
            try:
                # e.g. response = {
                #   'data': [
                #       {
                #           '_id': 'f1fc5d71-24a8-4b38-9c5b-0cba1e615acb', 
                #           'difference': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                #       }, 
                #       {   
                #           '_id': '0997b6f4-ec0c-49fc-8428-1824c496a964', 
                #           'difference': [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
                #       }
                #    ]
                # }
                difference_shares_party_i = response.json().get("data")
                if difference_shares_party_i is None:
                    raise ValueError(f"Error in Response: {response.text}")
                difference_shares.append(difference_shares_party_i)
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text

        return difference_shares
    

    def chunk_query_execute(self, chunk_ids):

        chunk_shares = []
        for node in self.nodes:
            url = node.url + "/data/read"
            # Authorization header with the provided token
            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }

            # Schema payload
            payload = {
                "schema": node.schema_id,
                "filter": {
                    "_id": {
                        "$in": chunk_ids
                    }
                }
            }

            # Send POST request
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise ValueError(f"Error in POST request: {response.status_code}, {response.text}")
            try:
                # e.g. response = {
                #   'data': [
                #       {
                #           '_id': 'f1fc5d71-24a8-4b38-9c5b-0cba1e615acb', 
                #           'difference': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                #       }, 
                #       {   
                #           '_id': '0997b6f4-ec0c-49fc-8428-1824c496a964', 
                #           'difference': [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
                #       }
                #    ]
                # }
                chunk_shares_party_i = response.json().get("data")
                if chunk_shares_party_i is None:
                    raise ValueError(f"Error in Response: {response.text}")
                chunk_shares.append(chunk_shares_party_i)
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text

        return chunk_shares



    # Function to store embedding and chunk in RAG database
    def upload_data(self, lst_embedding_shares, lst_chunk_shares):
        # lst_embeddings_shares [20][384][2]
        # lst_chunks_shares [20][2][268]

        # Check sizes: same number of embeddings and chunks
        assert len(lst_embedding_shares) == len(lst_chunk_shares), f"Mismatch: {len(lst_embedding_shares)} embeddings vs {len(lst_chunk_shares)} chunks."
        
        for (embedding_shares, chunk_shares) in zip(lst_embedding_shares, lst_chunk_shares):
            # embeddings_shares [384][2]
            # chunks_shares [2][268]
            
            # 'data_id' has to be the same for every node to allow secret reconstructions
            data_id = str(uuid4()) 
            for i, node in enumerate(self.nodes):
                url = node.url + "/data/create"
                # Authorization header with the provided token
                headers = {
                    "Authorization": node.bearer_token,
                    "Content-Type": "application/json"
                }
                # Join the shares of one embedding in one vector
                node_i_embedding_shares = [e[i] for e in embedding_shares]
                node_i_chunk_share = chunk_shares[i]
                # encode to be parsed in json
                encoded_node_i_chunk_share = base64.b64encode(node_i_chunk_share).decode('utf-8')
                # Schema payload
                payload = {
                    "schema": node.schema_id,
                    "data": [
                        {
                            "_id": data_id,
                            "embedding": node_i_embedding_shares,
                            "chunk": encoded_node_i_chunk_share
                        }
                    ]
                }

                # Send POST request
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                if response.status_code != 200:
                    raise ValueError(f"Error in POST request: {response.status_code}, {response.text}")
                else:
                    print(
                        {
                            "status_code": response.status_code,
                            "message": "Success",
                            "response_json": response.json()
                        }
                    )


if __name__ == "__main__":

    json_file = "nildb_config.json"

    # Load NilDB from JSON file if it exists
    if os.path.exists(json_file):
        print("Loading NilDB configuration from file...")
        nilDB = NilDB.from_json(json_file)
    else:
        # Initialize NilDB if no configuration file exists
        print("No configuration file found. Initializing NilDB...")
        nilDB_nodes = [
            # node-a50d
            Node(
                url="https://nildb-node-a50d.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
                owner="did:nil:testnet:nillion1qhquyt20eq0vutjzw6v6zk752y6my4krxcmnn2",
                bearer_token="Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI2NjUsImV4cCI6MTczNTA0MjY2NSwiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xNWxjanhnYWZndnM0MHJ5cHZxdTczZ2Z2eDZwa3g3dWdkamE1MGQiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9.6zscB_D2KH3qSx00yHMeRVU9xpj-J0wKEFRoJg5fmZJyDgtN55Fypd69YxLWaVtze6l848X_r29QOIPM__QEDw",
            ),
            # node-dvml
            Node(
                url="https://nildb-node-dvml.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
                owner="did:nil:testnet:nillion1qhquyt20eq0vutjzw6v6zk752y6my4krxcmnn2",
                bearer_token="Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI2MzMsImV4cCI6MTczNTA0MjYzMywiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xZGZoNDRjczRoMnplazV2aHp4a2Z2ZDl3MjhzNXE1Y2RlcGR2bWwiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9.drcbqEBwWvSCOJvGfLX1FlvIwiXt-vXT6NxTsgRjkbsGbPLKg94KVYr0mcggfLXaMwarkAVdTxfanOX3otu4Bw",
            ),
            # node-guue
            Node(
                url="https://nildb-node-guue.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
                owner="did:nil:testnet:nillion1qhquyt20eq0vutjzw6v6zk752y6my4krxcmnn2",
                bearer_token="Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI0MjYsImV4cCI6MTczNTA0MjQyNiwiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xOXQwZ2VmbTdwcjZ4amtxMnNqNDBmMHJzN3d6bmxkZ2ZnNGd1dWUiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9._67NLp51SPK1nk7y-nMhzsD67Rp3HwlRxvC9w82cHgQBzL2XpkynptTIEy7kLJifHeUhdbtstbiGfL6MDtIlQQ",
            ),
            # Add more nodes here if needed.
        ]
        nilDB = NilDB(nilDB_nodes)

        # Initialize schema and queries
        nilDB.init_schema()
        nilDB.init_diff_query()

        # Save NilDB to JSON file
        nilDB.to_json(json_file)
        print("NilDB configuration saved to file.")

    print("NilDB instance:", nilDB)
    print()

    # Set up modes of operation (i.e., keys)
    additive_key = nilql.secret_key({'nodes': [{}] * len(nilDB.nodes)}, {'sum': True})
    xor_key = nilql.secret_key({'nodes': [{}] * len(nilDB.nodes)}, {'store': True})

    # Create chunks nd embeddings
    file_path = 'data/cities.txt'
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)
    print('Generating embeddings...')
    embeddings = generate_embeddings_huggingface(chunks)
    print('Embeddings generated!')
    # print("Chunks: \n", chunks)
    # print("Chunks: \n", chunks[0])

    # print(f"Embeddings[{len(embeddings)}][{len(embeddings[0])}]")
    # print(f"chunks[{len(chunks)}][{len(chunks[0])}]")

    chunks_shares = []
    for chunk in chunks:
        chunks_shares.append(nilql.encrypt(xor_key, chunk))

    embeddings_shares = []
    for embedding in embeddings:
        embeddings_shares.append(encrypt_float_list(additive_key, embedding))

    # print(f"embeddings_shares [{len(embeddings_shares)}][{len(embeddings_shares[0])}][{len(embeddings_shares[0][0])}]")
    # print(f"chunks_shares [{len(chunks_shares)}][{len(chunks_shares[0])}][{len(chunks_shares[0][0])}]")

    # Upload data to nilDB
    print('Uploading data:')
    nilDB.upload_data(embeddings_shares, chunks_shares)
    

