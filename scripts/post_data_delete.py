# Posts the tee query

import requests
import json
from uuid import uuid4 # Generate a UUID4 (random-based UUID) _id = uuid4()
import numpy as np

# URL for the POST request
url = [
    "https://nildb-node-a50d.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/data/flush",
    "https://nildb-node-dvml.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/data/flush",
    "https://nildb-node-guue.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/data/flush",
]

# Authorization header with your token
headers = [
    {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI2NjUsImV4cCI6MTczNTA0MjY2NSwiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xNWxjanhnYWZndnM0MHJ5cHZxdTczZ2Z2eDZwa3g3dWdkamE1MGQiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9.6zscB_D2KH3qSx00yHMeRVU9xpj-J0wKEFRoJg5fmZJyDgtN55Fypd69YxLWaVtze6l848X_r29QOIPM__QEDw",
        "Content-Type": "application/json"
    },
    {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI2MzMsImV4cCI6MTczNTA0MjYzMywiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xZGZoNDRjczRoMnplazV2aHp4a2Z2ZDl3MjhzNXE1Y2RlcGR2bWwiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9.drcbqEBwWvSCOJvGfLX1FlvIwiXt-vXT6NxTsgRjkbsGbPLKg94KVYr0mcggfLXaMwarkAVdTxfanOX3otu4Bw",
        "Content-Type": "application/json"
    },
    {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NksifQ.eyJpYXQiOjE3MzQ0NDI0MjYsImV4cCI6MTczNTA0MjQyNiwiYXVkIjoiZGlkOm5pbDp0ZXN0bmV0Om5pbGxpb24xOXQwZ2VmbTdwcjZ4amtxMnNqNDBmMHJzN3d6bmxkZ2ZnNGd1dWUiLCJpc3MiOiJkaWQ6bmlsOnRlc3RuZXQ6bmlsbGlvbjFxaHF1eXQyMGVxMHZ1dGp6dzZ2NnprNzUyeTZteTRrcnhjbW5uMiJ9._67NLp51SPK1nk7y-nMhzsD67Rp3HwlRxvC9w82cHgQBzL2XpkynptTIEy7kLJifHeUhdbtstbiGfL6MDtIlQQ",
        "Content-Type": "application/json"
    },
]

# Schema payload
payload = {
        "schema": "6aa651af-7762-4aaa-9089-82f8eab16201"
}

for u, h in zip(url, headers):
    # Send POST request
    response = requests.post(u, headers=h, data=json.dumps(payload))

    # Print the response
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())




