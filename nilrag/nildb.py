import requests
import json
from uuid import uuid4
import os

class Node:
    def __init__(self, url, org, bearer_token):
        self.url = url[:-1] if url.endswith('/') else url
        self.org = str(org)
        self.bearer_token = str(bearer_token)
        self.schema_id = None
        self.query_id = None

    def __repr__(self):
        return f"URL: {self.url}\
            \nOrg: {self.org}\
            \nBearer Token: {self.bearer_token}\
            \nSchema ID: {self.schema_id}\
            \nQuery ID: {self.query_id}"

    def to_dict(self):
        """Convert Node to a dictionary for serialization."""
        return {
            "url": self.url,
            "org": self.org,
            "bearer_token": self.bearer_token,
            "schema_id": self.schema_id,
            "query_id": self.query_id
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Node instance from a dictionary."""
        node = cls(data["url"], data["org"], data["bearer_token"])
        node.schema_id = data.get("schema_id")
        node.query_id = data.get("query_id")
        return node

class NilDB:
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        for node in self.nodes:
            return node.__repr__()

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
        for node in self.nodes:
            url = node.url + "/schemas"

            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }
            payload = {
                "org": node.org,
                "name": "Nillion Users 4",
                "keys": ["_id"],
                "schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "NILLION USERS",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "_id": {
                                "type": "string",
                                "format": "uuid"
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
            try:
                schema_id = response.json().get("data")
                if schema_id is None:
                    raise ValueError(f"Error in Response: {response.text}")
                node.schema_id = schema_id
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text


    def init_query(self):
        # Send POST request
        for node in self.nodes:
            url = node.url + "/queries"

            headers = {
                "Authorization": node.bearer_token,
                "Content-Type": "application/json"
            }
            payload = {
                "org": node.org,
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
            try:
                query_id = response.json().get("data")
                if query_id is None:
                    raise ValueError(f"Error in Response: {response.text}")
                node.query_id = query_id
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text


    # Function to store embedding and chunk in RAG database
    def upload_data(self, embeddings_shares, chunks_shares):
        for node in self.nodes:
            url = node.url + "/data/create"
            bearer_token = node.bearer_token

            # Authorization header with the provided token
            headers = {
                "Authorization": bearer_token,
                "Content-Type": "application/json"
            }
            
            print("embeddings_shares", embeddings_shares)
            print("chunks", chunks_shares)
            # assert len(chunks) == len(embeddings), f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."

            # # For every chunk, save the corresponding embedding
            # for (embedding, chunk) in zip(embeddings, chunks):
            #     # Transform [Share] into [int]
            #     vector_of_int_embedding = [e.share for e in embedding]
            #     # Schema payload    
            #     payload = {
            #         "schema": schema_id,
            #         "data": [
            #             {
            #                 "_id": str(uuid4()),
            #                 "embedding": vector_of_int_embedding,
            #                 "chunk": chunk
            #             }
            #         ]
            #     }

            #     # Send POST request
            #     response = requests.post(url, headers=headers, data=json.dumps(payload))

            #     # Handle and return the response
            #     if response.status_code == 200:
            #         print( 
            #             {
            #                 "status_code": response.status_code,
            #                 "message": "Success",
            #                 "response_json": response.json()
            #             }
            #         )
            #     else:
            #         return  {
            #                 "status_code": response.status_code,
            #                 "message": "Failed to store data",
            #                 "response_json": response.json()
            #                 }


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
            Node(
                "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
                "b3d3f64d-ef12-41b7-9ff1-0e7681947bea",
                "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
            ),
        ]
        nilDB = NilDB(nilDB_nodes)

        # Initialize schema and query
        nilDB.init_schema()
        nilDB.init_query()

        # Save NilDB to JSON file
        nilDB.to_json(json_file)
        print("NilDB configuration saved to file.")

    print("NilDB instance:", nilDB)


