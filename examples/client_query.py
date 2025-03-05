"""
Example of querying nilDB with NilAI using nilRAG.
"""

import os
import sys
import json
from nilrag.nildb_requests import NilDB, Node


JSON_FILE = "examples/query_nildb_config.json"

# Load NilDB from JSON file if it exists
if os.path.exists(JSON_FILE):
    print("Loading NilDB configuration from file...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        nodes = []
        for node_data in data["nodes"]:
            nodes.append(
                Node(
                    url=node_data["url"],
                    node_id=None,
                    org=None,
                    bearer_token=node_data.get("bearer_token"),
                    schema_id=node_data.get("schema_id"),
                    diff_query_id=node_data.get("diff_query_id"),
                )
            )
        nilDB = NilDB(nodes)
else:
    print("Error: NilDB configuration file not found.")
    sys.exit(1)

print("NilDB instance:", nilDB)
print()

print('Query nilAI with nilRAG...')
response = nilDB.nilai_chat_completion(
    nilai_url="NILAI_URL", # Update with your nilAI node url
    token="NILAI_TOKEN", # Update with your nilAI token
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Tell me about Asia."}
    ],
    temperature=0.2,
    max_tokens=2048,
    stream=False,
)
print(response)
