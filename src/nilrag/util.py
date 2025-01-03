from sentence_transformers import SentenceTransformer
import numpy as np
import nilql

# Load text from file
def load_file(file_path):
    """
    Load text from a file and split it into paragraphs.

    Args:
        file_path (str): Path to the text file to load

    Returns:
        list: List of non-empty paragraphs with whitespace stripped
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split('\n\n')  # Split by double newline to get paragraphs
    return [para.strip() for para in paragraphs if para.strip()]  # Clean empty paragraphs


def create_chunks(paragraphs, chunk_size=500, overlap=100):
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
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


def generate_embeddings_huggingface(chunks_or_query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
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


def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two vectors.

    Args:
        a (array-like): First vector
        b (array-like): Second vector

    Returns:
        float: Euclidean distance between vectors a and b
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def find_closest_chunks(query_embedding, chunks, embeddings, top_k=2):
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

def group_shares_by_id(shares_per_party, transform_share_fn):
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
            id = share['_id']
            if id not in shares_by_id:
                shares_by_id[id] = []
            shares_by_id[id].append(transform_share_fn(share))
    return shares_by_id


PRECISION = 7
SCALING_FACTOR = 10 ** PRECISION


def to_fixed_point(value):
    """
    Convert a floating-point value to fixed-point representation.

    Args:
        value (float): Value to convert

    Returns:
        int: Fixed-point representation with PRECISION decimal places
    """
    return int(round(value * SCALING_FACTOR))


def from_fixed_point(value):
    """
    Convert a fixed-point value back to floating-point.

    Args:
        value (int): Fixed-point value to convert

    Returns:
        float: Floating-point representation
    """
    return value / SCALING_FACTOR


def encrypt_float_list(sk, lst):
    """
    Encrypt a list of floats using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of float values to encrypt

    Returns:
        list: List of encrypted fixed-point values
    """
    return [nilql.encrypt(sk, to_fixed_point(l)) for l in lst]


def decrypt_float_list(sk, lst):
    """
    Decrypt a list of encrypted fixed-point values to floats.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted fixed-point values

    Returns:
        list: List of decrypted float values
    """
    return [from_fixed_point(nilql.decrypt(sk, l)) for l in lst]


def encrypt_string_list(sk, lst):
    """
    Encrypt a list of strings using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of strings to encrypt

    Returns:
        list: List of encrypted strings
    """
    return [nilql.encrypt(sk, l) for l in lst]


def decrypt_string_list(sk, lst):
    """
    Decrypt a list of encrypted strings.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted strings

    Returns:
        list: List of decrypted strings
    """
    return [nilql.decrypt(sk, l) for l in lst]


# if __name__ == "__main__":
#     secret_key = "b1f6a40ae05a69d8fefd43af420b5ecb1a75e736eb2cce3d34eebfe9b45fb688"
#     org_did = "did:nil:testnet:nillion12d545xtad899pqp6xzvvnwqdkwlz0klysxljzn"
#     node_ids = [
#         "did:nil:testnet:nillion15lcjxgafgvs40rypvqu73gfvx6pkx7ugdja50d",
#         "did:nil:testnet:nillion17bkjqvcqyfjdnf04hfztrh9rfkj9qfjlzjqvn2",
#         "did:nil:testnet:nillion18zmcgyfjqz94lq7tfd8w4qvxdw99jfdmznd7hv"
#     ]
#     generate_jwt(secret_key, org_did, node_ids)


# nilrag.generate_jwt(secret_key, org, ttl, node_config)
#     Generates and outputs new jwt
#     Stores it in nilDB nodes
# nilrag.query(jwt, query_string, schema_id, query_id, tee_config, node_config)
#     Generates json request for chat completion endpoint
#     Calls chat completion endpoint (link with nilAI/TEE)
