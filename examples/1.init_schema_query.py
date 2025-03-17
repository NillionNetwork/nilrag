"""
Script to initialize schema and query to nilDB.
"""

import os
import json
import sys
from nilrag.nildb_requests import NilDB, Node


JSON_FILE = "examples/nildb_config.json"

# Load NilDB from JSON file if it exists
secret_key = None
if os.path.exists(JSON_FILE):
    print("Loading NilDB configuration from file...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        secret_key = data["org_secret_key"]
        nodes = []
        for node_data in data["nodes"]:
            nodes.append(
                Node(
                    node_data["url"],
                    node_data["node_id"],
                    data["org_did"],
                )
            )
        nilDB = NilDB(nodes)
else:
    print("Error: NilDB configuration file not found.")
    sys.exit(1)

jwts = nilDB.generate_jwt(secret_key, ttl=3600)

print("NilDB instance:", nilDB)
print()

# Upload encrypted data to nilDB
print("Initializing schema...")
schema_id = nilDB.init_schema()
print("Schema initialized successfully")

print("Initializing query...")
diff_query_id = nilDB.init_diff_query()
print("Query initialized successfully")

with open(JSON_FILE, "w", encoding="utf-8") as f:
    for node_data, jwt in zip(data["nodes"], jwts):
        node_data["schema_id"] = schema_id
        node_data["diff_query_id"] = diff_query_id
        node_data["bearer_token"] = jwt
    json.dump(data, f, indent=4)
print("Updated nilDB configuration file with schema and query IDs.")
