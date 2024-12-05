import requests
import json

# URL for the POST request
url = "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/schemas"

# Authorization header with your token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
    "Content-Type": "application/json"
}

# Schema payload
payload = {
    "org": "b3d3f64d-ef12-41b7-9ff1-0e7681947bea",
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

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())