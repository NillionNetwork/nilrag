from sentence_transformers import SentenceTransformer
import numpy as np
from secret_sharing import *
from phe import paillier
import time

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

# Generate embeddings using HuggingFace
def generate_embeddings_huggingface(chunks_or_query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings

# Calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Find the closest chunks to a query
def find_closest_chunks(query_embedding, chunks, embeddings, top_k=2):
    distances = [euclidean_distance(query_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(distances)
    return [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]


def rag_plaintext():
    file_path = "data.txt"
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)  # Reduced chunk size for clarity
    embeddings = generate_embeddings_huggingface(chunks)

    # User Query
    query = "Tell me about places in Asia."
    query_embedding = generate_embeddings_huggingface([query])[0]

    # Retrieve top results
    top_results = find_closest_chunks(query_embedding, chunks, embeddings)

    # Display results
    print("Query:", query)
    print("\nTop Matches:")
    for i, (chunk, dist) in enumerate(top_results, 1):
        print(f"{i}. {chunk} (Distance: {dist:.2f})")

def rag_with_shares():
    precision = 7
    secret_sharing = AdditiveSecretSharing(3, 2**31 - 1, precision)

    file_path = "data.txt"
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)  # Reduced chunk size for clarity
    embeddings = generate_embeddings_huggingface(chunks)
    all_embedding_shares = []
    for embedding_share in embeddings:
        all_embedding_shares.append(secret_sharing.secret_share(embedding_share))
    assert len(all_embedding_shares[0]) == secret_sharing.num_parties

    # User Query
    query = "Tell me about places in Asia."
    query_embedding = generate_embeddings_huggingface([query])[0]
    query_embedding_shares = secret_sharing.secret_share(query_embedding)
    assert len(query_embedding_shares) == secret_sharing.num_parties

    # Compute the differences between the query and all embeddings
    all_differences_shares = []
    for embedding_share in all_embedding_shares:
        all_differences_shares.append(
            [np.array(query_embedding_shares[party]) - np.array(embedding_share[party])
            for party in range(secret_sharing.num_parties)]
        )

    # Reveal
    differences = []
    for differences_share in all_differences_shares:
        differences.append(secret_sharing.secret_reveal(differences_share))

    # Compute Euclidean distances based on differences
    distances = [np.linalg.norm(diff) for diff in differences]
    # Retrieve top results
    sorted_indices = np.argsort(distances)
    top_k = 2
    top_results = [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]

    # Display results
    print("Query:", query)
    print("\nTop Matches:")
    for i, (chunk, dist) in enumerate(top_results, 1):
        print(f"{i}. {chunk} (Distance: {dist:.2f})")


def rag_with_paillier():
    public_key, private_key = paillier.generate_paillier_keypair()

    file_path = "data.txt"
    paragraphs = load_file(file_path)
    chunks = create_chunks(paragraphs, chunk_size=50, overlap=10)  # Reduced chunk size for clarity
    embeddings = generate_embeddings_huggingface(chunks)
    all_embedding_shares = []
    for i, embedding_share in enumerate(embeddings):
        t = []
        for j, e in enumerate(embedding_share):
            t = public_key.encrypt(float(e))
            print(f'finished {j} out of {len(embedding_share)}')
        all_embedding_shares.append(t)
        print(f'finished {i} out of {len(embeddings)}')
    print('finished encryption of documents')

    # User Query
    query = "Tell me about places in Asia."
    query_embedding = generate_embeddings_huggingface([query])[0]
    query_embedding_shares = [public_key.encrypt(float(q)) for q in query_embedding]
    print('finished encryption of query')

    # Compute the differences between the query and all embeddings
    all_differences_shares = []
    for embedding_share in all_embedding_shares:
        all_differences_shares.append(
            np.array(query_embedding_shares) - np.array(embedding_share)
        )
    print('Computed difference')

    # Reveal
    differences = []
    for differences_share in all_differences_shares:
        differences.append([private_key.decrypt(d) for d in differences_share])

    # Compute Euclidean distances based on differences
    distances = [np.linalg.norm(diff) for diff in differences]
    # Retrieve top results
    sorted_indices = np.argsort(distances)
    top_k = 2
    top_results = [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]

    # Display results
    print("Query:", query)
    print("\nTop Matches:")
    for i, (chunk, dist) in enumerate(top_results, 1):
        print(f"{i}. {chunk} (Distance: {dist:.2f})")


if __name__ == "__main__":
    secret_sharing_test()
    print()

    start_time = time.time()
    rag_plaintext()
    print("[Plaintext] Elapsed time:", time.time() - start_time)
    print()

    start_time = time.time()
    rag_with_shares()
    print("[Secret Shares] Elapsed time:", time.time() - start_time)
    print()

    # start_time = time.time()
    # rag_with_paillier()
    # print("[Paillier] Elapsed time:", time.time() - start_time)
    # print()
