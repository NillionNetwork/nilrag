import requests
import json
from uuid import uuid4 # Generate a UUID4 (random-based UUID) _id = uuid4()
import numpy as np

# URL for the POST request
url = "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/data/create"

# Authorization header with your token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
    "Content-Type": "application/json"
}

# Schema payload
payload = {
    "schema": "4f96c43a-41b0-4aa2-8c42-729688e20c4a",
    "data": [
        {
            "_id": str(uuid4()),
            "embedding": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chunk": "This is the first text we insert"
        },
        {
            "_id": str(uuid4()),
            "embedding": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "chunk": "This is the second text we insert"
        }
    ]
}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())








