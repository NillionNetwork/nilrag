# FastAPI and serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import argparse
# from crypto.secret_sharing import *
import base64
import logging
from nilrag.nildb import NilDB
from nilrag.util import generate_embeddings_huggingface, encrypt_float_list, decrypt_float_list, group_shares_by_id
import os
import nilql
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add config settings model
class Settings:
    def __init__(self, config_path: str = None):
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Launch TEE server with config')
        parser.add_argument('--config', '-c', 
                        default="examples/tee_nildb_config.json",
                        help='Path to nilDB config file')
        
        args = parser.parse_args()
        self.config_path = args.config

# Create settings instance
settings = Settings()

# Create app with lifespan to handle startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config on startup
    if not os.path.exists(settings.config_path):
        raise Exception(f"No nildb config file found at: {settings.config_path}")
    app.state.nildb = NilDB.from_json(settings.config_path)
    yield
    # Clean up on shutdown if needed
    pass

app = FastAPI(
    title="NilRAG",
    description="A RAG platform powered by secure, confidential computing.",
    version="0.1.0",
    terms_of_service="https://nillion.com",
    contact={
        "name": "Nillion AI Support",
        "email": "research@nillion.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    lifespan=lifespan
)

# Define a model for the query payload
class Query(BaseModel):
    query: str  # The query sent by the client for LLM inference

# Add this after the Query class
class Document(BaseModel):
    content: str
    document_id: str | None = None  # Optional ID, will be auto-generated if not provided

@app.post("/ingest-document")
async def ingest_document(document: Document):
    """
    Endpoint to ingest a document into the system.
    1. Generate embeddings for the document
    2. Secret share the document and embeddings
    3. Store in NilDB
    """

@app.post("/process-client-query")
async def process_client_query(payload: Query):
    """
    Endpoint to process a client query.
    1. Initialization: Secret share keys and NilDB instance.
    2. Secret share query and send to NilDB.
    3. Ask NilDB to compute the differences.
    4. Compute distances and sort.
    5. Ask NilDB to return top k chunks.
    6. Append top results to LLM query
    """
    try:
        logger.info("Received query: %s", payload.query)

        # Step 1: Initialization
        # Get NilDB instance from app state
        nilDB = app.state.nildb
        # Initialize secret keys
        num_parties = len(nilDB.nodes)
        additive_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'sum': True})
        xor_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'store': True})


        # Step 2: Secret share query
        logger.debug("Secret sharing query and sending to NilDB...")
        # 2.1 Generate query embeddings: one string query is assumed.
        query_embedding = generate_embeddings_huggingface([payload.query])[0]
        nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)


        # Step 3: Ask NilDB to compute the differences
        logger.debug("Requesting computation from NilDB...")
        difference_shares = nilDB.diff_query_execute(nilql_query_embedding)


        # Step 4: Compute distances and sort
        logger.debug("Compute distances and sort...")
        # 4.1 Group difference shares by ID
        difference_shares_by_id = group_shares_by_id(
            difference_shares,
            lambda share: share['difference']
        )
        # 4.2 Transpose the lists for each _id
        difference_shares_by_id = {
            id: np.array(differences).T.tolist()
            for id, differences in difference_shares_by_id.items()
        }
        # 4.3 Decrypt and compute distances
        reconstructed = [
            {
                '_id': id, 
                'distances': np.linalg.norm(decrypt_float_list(additive_key, difference_shares))
            } 
            for id, difference_shares in difference_shares_by_id.items()
        ]
        # 4.4 Sort id list based on the corresponding distances
        sorted_ids = sorted(reconstructed, key=lambda x: x['distances'])


        # Step 5: Query the top k 
        logger.debug("Query top k chunks...")
        top_k = 2
        top_k_ids = [item['_id'] for item in sorted_ids[:top_k]]
        
        # 5.1 Query top k
        chunk_shares = nilDB.chunk_query_execute(top_k_ids)
        
        # 5.2 Group chunk shares by ID
        chunk_shares_by_id = group_shares_by_id(
            chunk_shares,
            lambda share: base64.b64decode(share['chunk'])
        )

        # 5.3 Decrypt chunks
        top_results = [
            {
                '_id': id, 
                'distances': nilql.decrypt(xor_key, chunk_shares)
            } 
            for id, chunk_shares in chunk_shares_by_id.items()
        ]

        # Step 6: Append top results to LLM query
        logger.debug(" Append top results to LLM query...")
        formatted_results = "\n".join(
            f"- {str(result['distances'])}" for result in top_results
        )
        llm_query = payload.query + "\n\nRelevant Context:\n" + formatted_results


        # Placeholder for final response
        final_response = {"RAG output": llm_query}
        logger.info("Successfully processed query. Returning response.")

        return final_response

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

