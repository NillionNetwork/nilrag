import os
from nilrag.nildb import NilDB


# Initialize NilDB
print("Initializing NilDB...")
uninitialized_json_file = "uninitialized_nildb_config.json"
initialized_json_file = "initialized_nildb_config.json"

# Load NilDB from JSON file if it exists
if os.path.exists(uninitialized_json_file):
    print("Loading NilDB configuration from file...")
    nilDB = NilDB.from_json(uninitialized_json_file)
else:
    print("Error: NilDB configuration file not found.")
    exit(1)

# Initialize schema and queries
nilDB.init_schema()
nilDB.init_diff_query()

# Save NilDB to JSON file
nilDB.to_json(initialized_json_file)
print("NilDB configuration saved to file.")

