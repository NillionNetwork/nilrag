# Posts the tee query

import requests
import json
from uuid import uuid4 # Generate a UUID4 (random-based UUID) _id = uuid4()
import numpy as np

# URL for the POST request
url = "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/queries"

# Authorization header with your token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
    "Content-Type": "application/json"
}

# Schema payload
payload = {
        "org": "b3d3f64d-ef12-41b7-9ff1-0e7681947bea",
        "name": "Returns the difference between the nilDB embeddings and the query embedding",
        "schema": "3465b238-895c-49ba-9399-433d1491bb7a",
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

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())




