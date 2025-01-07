"""
Utility functions for nilRAG.
"""

from typing import Union

import nilql
import numpy as np
from sentence_transformers import SentenceTransformer


# Load text from file
def load_file(file_path: str):
    """
    Load text from a file and split it into paragraphs.

    Args:
        file_path (str): Path to the text file to load

    Returns:
        list: List of non-empty paragraphs with whitespace stripped
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = text.split("\n\n")  # Split by double newline to get paragraphs
    return [
        para.strip() for para in paragraphs if para.strip()
    ]  # Clean empty paragraphs


def create_chunks(paragraphs: list[str], chunk_size: int = 500, overlap: int = 100):
    """
    Split paragraphs into overlapping chunks of words.

    Args:
        paragraphs (list): List of paragraph strings to chunk
        chunk_size (int, optional): Maximum number of words per chunk. Defaults to 500.
        overlap (int, optional): Number of overlapping words between chunks. Defaults to 100.

    Returns:
        list: List of chunk strings with specified size and overlap
    """
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
    return chunks


def generate_embeddings_huggingface(
    chunks_or_query: Union[str, list],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Generate embeddings for text using a HuggingFace sentence transformer model.

    Args:
        chunks_or_query (str or list): Text string(s) to generate embeddings for
        model_name (str, optional): Name of the HuggingFace model to use.
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        numpy.ndarray: Array of embeddings for the input text
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings


def euclidean_distance(a: list, b: list):
    """
    Calculate Euclidean distance between two vectors.

    Args:
        a (array-like): First vector
        b (array-like): Second vector

    Returns:
        float: Euclidean distance between vectors a and b
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def find_closest_chunks(
    query_embedding: list, chunks: list, embeddings: list, top_k: int = 2
):
    """
    Find chunks closest to a query embedding using Euclidean distance.

    Args:
        query_embedding (array-like): Embedding vector of the query
        chunks (list): List of text chunks
        embeddings (list): List of embedding vectors for the chunks
        top_k (int, optional): Number of closest chunks to return. Defaults to 2.

    Returns:
        list: List of tuples (chunk, distance) for the top_k closest chunks
    """
    distances = [euclidean_distance(query_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(distances)
    return [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]


def group_shares_by_id(shares_per_party: list, transform_share_fn: callable):
    """
    Groups shares by their ID and applies a transform function to each share.

    Args:
        shares_per_party (list): List of shares from each party
        transform_share_fn (callable): Function to transform each share value

    Returns:
        dict: Dictionary mapping IDs to list of transformed shares
    """
    shares_by_id = {}
    for party_shares in shares_per_party:
        for share in party_shares:
            share_id = share["_id"]
            if share_id not in shares_by_id:
                shares_by_id[share_id] = []
            shares_by_id[share_id].append(transform_share_fn(share))
    return shares_by_id


PRECISION = 7
SCALING_FACTOR = 10**PRECISION


def to_fixed_point(value: float) -> int:
    """
    Convert a floating-point value to fixed-point representation.

    Args:
        value (float): Value to convert

    Returns:
        int: Fixed-point representation with PRECISION decimal places
    """
    return int(round(value * SCALING_FACTOR))


def from_fixed_point(value: int) -> float:
    """s
    Convert a fixed-point value back to floating-point.

    Args:
        value (int): Fixed-point value to convert

    Returns:
        float: Floating-point representation
    """
    return value / SCALING_FACTOR


def encrypt_float_list(sk, lst: list[float]) -> list[list]:
    """
    Encrypt a list of floats using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of float values to encrypt

    Returns:
        list: List of encrypted fixed-point values
    """
    return [nilql.encrypt(sk, to_fixed_point(l)) for l in lst]


def decrypt_float_list(sk, lst: list[list]) -> list[float]:
    """
    Decrypt a list of encrypted fixed-point values to floats.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted fixed-point values

    Returns:
        list: List of decrypted float values
    """
    return [from_fixed_point(nilql.decrypt(sk, l)) for l in lst]


def encrypt_string_list(sk, lst: list) -> list:
    """
    Encrypt a list of strings using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of strings to encrypt

    Returns:
        list: List of encrypted strings
    """
    return [nilql.encrypt(sk, l) for l in lst]


def decrypt_string_list(sk, lst: list) -> list:
    """
    Decrypt a list of encrypted strings.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted strings

    Returns:
        list: List of decrypted strings
    """
    return [nilql.decrypt(sk, l) for l in lst]
