# FastAPI and serving
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
# from crypto.secret_sharing import *
import logging
from .nildb import NilDB
from .util import generate_embeddings_huggingface, encrypt_float_list, decrypt_float_list
import os
import nilql

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# host = "nilrag.sandbox.nilogy.xyz"

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
    # openapi_tags=[],
)

# Define a model for the query payload
class Query(BaseModel):
    query: str  # The query sent by the client for LLM inference

@app.post("/process-client-query")
async def process_client_query(payload: Query):
    """
    Endpoint to process a client query.
    1. Secret share query and send to NilDB.
    2. Ask NilDB to compute the differences.
    3. Wait for response.
    4. Find top k.
    5. Ask NilDB to return top k chunks
    6. Run LLM.
    7. Return answer to the user.
    """
    try:
        logger.info("Received query: %s", payload.query)

        # Step 1: Initialization:
        logger.debug("Creating NilDB instance...")
        # 1.1 Load NilDB from JSON file if it exists
        json_file = "nildb_config.json"
        if os.path.exists(json_file):
            print("Loading NilDB configuration from file...")
            nilDB = NilDB.from_json(json_file)
        else:
            raise Exception("No nildb config file found in the TEE server.")
        # 1.2 Initialize secret keys
        num_parties = len(nilDB.nodes)
        additive_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'sum': True})
        xor_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'store': True})


        # Step 2: Secret share query and send to NilDB
        logger.debug("Secret sharing query and sending to NilDB...")
        # 2.1 Generate query embeddings: one string query is assumed.
        query_embedding = generate_embeddings_huggingface([payload.query])[0]
        nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)


        # Step 3: Ask NilDB to compute the differences
        logger.debug("Requesting computation from NilDB...")
        # 3.1 Rearrange query_embedding_shares to group by party
        query_embedding_shares = [
            [entry[party] for entry in nilql_query_embedding]
            for party in range(num_parties)
        ]
        # 3.2 Ask NilDB to compute the differences
        difference_shares = nilDB.diff_query_execute(query_embedding_shares)


        # Step 4: Compute distances and sort
        logger.debug("Compute distances and sort...")
        # 4.1 Restructure the array of differences so it can be revealed next by nilQL.
        difference_shares_by_id = {}
        for difference_shares_per_party in difference_shares:
            for share in difference_shares_per_party:
                id = share['_id']
                # Add the difference list to the corresponding _id
                if id not in difference_shares_by_id:
                    difference_shares_by_id[id] = []
                difference_shares_by_id[id].append(share['difference']) 
        # 4.2 Decrypt and compute distances
        reconstructed = [
            {
                '_id': id, 
                'distances': np.linalg.norm(decrypt_float_list(additive_key, difference_shares))
            } 
            for id, difference_shares in difference_shares_by_id.items()
        ]
        # 4.3 Sort id list based on the corresponding distances
        # Assuming reconstructed is already defined
        sorted_ids = sorted(reconstructed, key=lambda x: x['distances'])



        # Step 5: Query the top k 
        logger.debug("Query top k chunks...")
        top_k = 2
        top_k_ids = [item['_id'] for item in sorted_ids[:top_k]]
        # 5.1 Query top k
        chunk_shares = nilDB.chunk_query_execute(top_k_ids)
        # 5.2 Reconstruct top k
        chunk_shares_by_id = {}
        for chunk_shares_per_party in chunk_shares:
            for share in chunk_shares_per_party:
                id = share['_id']
                # Add the chunk list to the corresponding _id
                if id not in chunk_shares_by_id:
                    chunk_shares_by_id[id] = []
                chunk_shares_by_id[id].append(share['chunk']) 
        # 4.2 Decrypt and compute distances
        top_results = [
            {
                '_id': id, 
                'distances': nilql.decrypt(xor_key, chunk_shares)
            } 
            for id, chunk_shares in chunk_shares_by_id.items()
        ]

        # Step 6: Append top results to LLM query
        logger.debug("Query top k chunks...")
        formatted_results = "\n".join(
            f"- {str(result['distances'])}" for result in top_results
        )
        llm_query = payload.query + "\n\nRelevant Context:\n" + formatted_results


        # Placeholder for final response
        final_response = {"response": "Placeholder for RAG/LLM output"}
        logger.info("Successfully processed query. Returning response.")

        return final_response

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

