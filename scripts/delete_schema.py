import requests

# URL for the DELETE request
url = "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/schemas"

# Authorization header with your token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
    "Content-Type": "application/json"
}

# Optional payload for DELETE (if the API requires it, adjust accordingly)
payload = {
    "id": "5b5ef0f7-af03-4aea-a7ee-aacf055d99cc"
}

# Send DELETE request
response = requests.delete(url, headers=headers, json=payload)

# Print the response
print("Status Code:", response.status_code)
if response.text:
    print("Response JSON:", response.json())
else:
    print("No response body.")