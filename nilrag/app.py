# FastAPI and serving
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from crypto.secret_sharing import *
import logging

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

        # Step 1: Secret share query and send to NilDB
        logger.debug("Secret sharing query and sending to NilDB...")
        query_embeddings = generate_embeddings_huggingface(payload.query)
        # secret_shared_query = secret_share(query_embeddings)
        # for query_party in secret_shared_query:
            # pass
            # send_to_nildb(query_party)

        # Step 2: Ask NilDB to compute the differences
        logger.debug("Requesting computation from NilDB...")
        # Ask NilDB to compute the differences

        # Step 3: Wait for response
        logger.debug("Waiting for response from NilDB...")
        # differences = wait_for_response()
        # if not computation_result:
        #     raise HTTPException(status_code=500, detail="NilDB computation failed")

        # Step 4: Run remaining steps of RAG and LLM
        logger.info("Running RAG ...")
        # distances = [np.linalg.norm(diff) for diff in differences]
        # sorted_indices = np.argsort(distances)
        # return [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]


        # Placeholder for final response
        final_response = {"response": "Placeholder for RAG/LLM output"}
        logger.info("Successfully processed query. Returning response.")

        return final_response

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Generate embeddings using HuggingFace
def generate_embeddings_huggingface(chunks_or_query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings
