import requests
import json

class nilDB_MPC:

    def __init__(self, cluster_URLs, orgs, bearer_tokens):
        assert len(cluster_URLs) == len(orgs) == len(bearer_tokens), "Number of cluster URLs, orgs, and bearer tokens should be the same"
        self.cluster_URLs = [url[:-1] if url.endswith('/') else url for url in cluster_URLs]
        self.orgs = orgs
        self.bearer_tokens = bearer_tokens
        self.cluster_size = len(cluster_URLs)

    def init_schema(self):
        # Send POST request
        for node in range(self.cluster_size):
            url = self.cluster_URLs[node] + "/schemas"
            org = self.orgs[node]
            bearer_tokens = self.bearer_tokens[node]

            headers = {
                "Authorization": str(bearer_tokens),
                "Content-Type": "application/json"
            }
            payload = {
                "org": str(org),
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
                data = response.json().get("data")
                if data is None:
                    raise ValueError(f"Error in Response: {response.text}")
                return data
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text


    def init_query(self, schema_id):
        # Send POST request
        for node in range(self.cluster_size):
            url = self.cluster_URLs[node] + "/queries"
            org = self.orgs[node]
            bearer_tokens = self.bearer_tokens[node]

            headers = {
                "Authorization": str(bearer_tokens),
                "Content-Type": "application/json"
            }
            payload = {
                "org": str(org),
                "name": "Returns the difference between the nilDB embeddings and the query embedding",
                "schema": str(schema_id),
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
                data = response.json().get("data")
                if data is None:
                    raise ValueError(f"Error in Response: {response.text}")
                return data
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                return response.text


if __name__ == "__main__":
    # URL for the POST request
    cluster_URLs = [
        "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
        # put here nodes 2-N
    ]
    orgs = [
        "b3d3f64d-ef12-41b7-9ff1-0e7681947bea",
        # put here nodes 2-N
    ]
    bearer_tokens = [
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
        # put here nodes 2-N
    ]

    nilDB = nilDB_MPC(cluster_URLs, orgs, bearer_tokens)    
    schema_id = nilDB.init_schema()
    print("Schema ID:", schema_id)

    query_id = nilDB.init_query(schema_id)
    print("Query ID:", query_id)
