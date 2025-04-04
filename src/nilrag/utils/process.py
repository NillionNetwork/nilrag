"""
This module provides functions to process embeddings and chunks.
"""

from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .transform import to_fixed_point


def check_inputs_to_upload(
    embedding_shares: list,
    chunk_shares: list,
    labels: list | None,
    centroids: list | None,
) -> None:
    """
    Checks if the number of embeddings and chunks is the same
    when there is clustering, check if each embedding was assigned a label.

    """
    # Check sizes: same number of embeddings and chunks
    assert len(embedding_shares) == len(
        chunk_shares
    ), f"Mismatch: {len(embedding_shares)} embeddings vs {len(chunk_shares)} chunks."
    # Check that the number of labels of each cluster matches the number of embeddings,
    # if labels are provided
    if labels is not None:
        assert len(labels) == len(
            embedding_shares
        ), f"Mismatch: {len(labels)} labels vs {len(embedding_shares)} embeddings."
    # Check that the number of centroids is correct if provided
    if centroids is not None:
        assert (
            len(centroids) > 1
        ), "Centroids must be provided when clustering is enabled."


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


def cluster_embeddings(embeddings: np.ndarray, num_clusters: int):
    """
    Cluster the given embeddings using K-Means.

    Args:
        embeddings (list of list of float): The embeddings to cluster.
        num_clusters (int): The number of clusters to form.

    Returns:
        tuple: (labels, centroids)
    """
    embeddings_array = np.array(embeddings)  # Convert to NumPy array
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_array)
    centroids = kmeans.cluster_centers_

    # Convert each centroid to fixed-point
    centroids = [[to_fixed_point(val) for val in centroid] for centroid in centroids]
    return labels, centroids
