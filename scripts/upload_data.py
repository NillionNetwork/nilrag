import requests
import json
from uuid import uuid4
from sentence_transformers import SentenceTransformer  # Ensure the library is installed
from crypto.secret_sharing import *

# Function to generate embeddings using Hugging Face
def generate_embeddings_huggingface(chunks_or_query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings

# Load text from file
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split('\n\n')  # Split by double newline to get paragraphs
    return [para.strip() for para in paragraphs if para.strip()]  # Clean empty paragraphs

# Chunk paragraphs into smaller pieces with overlap
def create_chunks(paragraphs, chunk_size=500, overlap=100):
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


# Function to store embedding and chunk in RAG database
def rag_fe_store_embedding_and_chunk(embeddings, chunks, url, schema, bearer_token):

    # Authorization header with the provided token
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }

    assert len(chunks) == len(embeddings), f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."

    # For every chunk, save the corresponding embedding
    for (embedding, chunk) in zip(embeddings, chunks):
        # Transform [Share] into [int]
        vector_of_int_embedding = [e.share for e in embedding]
        # Schema payload
        payload = {
            "schema": schema,
            "data": [
                {
                    "_id": str(uuid4()),
                    "embedding": vector_of_int_embedding,
                    "chunk": chunk
                }
            ]
        }

        # Send POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Handle and return the response
        if response.status_code == 200:
            print(
                {
                    "status_code": response.status_code,
                    "message": "Success",
                    "response_json": response.json()
                }
            )
        else:
            return  {
                    "status_code": response.status_code,
                    "message": "Failed to store data",
                    "response_json": response.json()
                    }



if __name__ == "__main__":
    # Config info
    schema = "162ff74d-6614-48a3-bb61-494ff95326b5"
    bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0"  # Replace with your token
    url = "https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/data/create"

    # Local secret sharing.
    precision = 7
    secret_sharing = AdditiveSecretSharing(2, 2**31 - 1, precision)

    # Read from file
    file_path = "data.txt"
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)  # Reduced chunk size for clarity
    # Create embeddings
    embeddings = generate_embeddings_huggingface(chunks)
    # Create shares
    all_embedding_shares = []
    for embedding_share in embeddings:
        all_embedding_shares.append(secret_sharing.secret_share(embedding_share))
    assert len(all_embedding_shares[0]) == secret_sharing.num_parties

    for party in range(secret_sharing.num_parties):
        # Store shares to each server
        array = np.array(all_embedding_shares)
        all_embedding_shares_for_party = array[:, party] # shares for party

        # Stores the embedding shares of party
        # Note: in the real deployment we need to change the url, schema and bearer_token for each database
        result = rag_fe_store_embedding_and_chunk(all_embedding_shares_for_party, chunks, url, schema, bearer_token)

        print(result)