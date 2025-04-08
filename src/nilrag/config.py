"""
Configuration loading utilities for nilRAG.
"""

import json
import os
from typing import Optional, Tuple

from dotenv import load_dotenv
_ = load_dotenv()

from nilrag.nildb_requests import NilDB, Node

def load_nil_db_config(
    config_path: str,
    require_secret_key: bool = False,
    require_bearer_token: bool = False,
    require_schema_id: bool = False,
    require_diff_query_id: bool = False,
) -> Tuple[NilDB, Optional[str]]:
    """
    Load nilDB configuration from JSON file.

    Args:
        config_path: Path to the configuration file
        require_secret_key: Whether to require org_secret_key in the config
        require_bearer_token: Whether to require bearer_token in node data
        require_schema_id: Whether to require schema_id in node data
        require_diff_query_id: Whether to require diff_query_id in node data

    Returns:
        tuple: (NilDB instance, secret_key if required)

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is invalid or required fields are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: NilDB configuration file not found at {config_path}")

    print(f"Loading NilDB configuration from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(f'Error: Invalid JSON in configuration file {config_path}') from exc

    # Get secret key if required
    secret_key = os.getenv("NILDB_ORG_SECRET_KEY")
    if require_secret_key and not secret_key:
        raise ValueError("Error: org_secret_key not found in NILDB_ORG_SECRET_KEY envvar")

    # Get org DID
    org_did = os.getenv("NILDB_ORG_DID")

    if not org_did:
        raise ValueError("Error: org_did not found in NILDB_ORG_DID envvar")

    # Create nodes
    nodes = []
    for node_data in data["nodes"]:
        # Validate required fields
        if require_bearer_token and "bearer_token" not in node_data:
            raise ValueError("Error: bearer_token not found in node data")
        if require_schema_id and "schema_id" not in node_data:
            raise ValueError("Error: schema_id not found in node data")
        if require_diff_query_id and "diff_query_id" not in node_data:
            raise ValueError("Error: diff_query_id not found in node data")

        # Create node with all available fields
        node = Node(
            url=node_data["url"],
            node_id=node_data.get("node_id"),
            org=org_did,
            bearer_token=node_data.get("bearer_token"),
            schema_id=node_data.get("schema_id"),
            diff_query_id=node_data.get("diff_query_id"),
        )
        nodes.append(node)

    return NilDB(nodes), secret_key
