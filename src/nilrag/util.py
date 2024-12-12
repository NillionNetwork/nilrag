from sentence_transformers import SentenceTransformer
import numpy as np
import nilql

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


PRECISION = 7
SCALING_FACTOR = 10 ** PRECISION


# Convert a floating-point value to fixed-point.
def to_fixed_point(value):
    return int(round(value * SCALING_FACTOR))


# Convert a fixed-point value back to floating-point.
def from_fixed_point(value):
    return value / SCALING_FACTOR


def encrypt_float_list(sk, lst):
    return [nilql.encrypt(sk, to_fixed_point(l)) for l in lst]


def encrypt_string_list(sk, lst):
    return [nilql.encrypt(sk, l) for l in lst]
