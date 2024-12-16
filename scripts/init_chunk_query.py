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
        "schema": "4f96c43a-41b0-4aa2-8c42-729688e20c4a",
        "variables": {
            "chunk_id": {
                "description": "The chunk id of interest",
                "type": "oid"
            }
        },
        "pipeline": [
    {
        "$addFields": {
            "chunk_id": "##chunk_id"
        }
    },
    {
        "$match": {
            "_id": "##chunk_id"
        }
    }
    ]
}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

