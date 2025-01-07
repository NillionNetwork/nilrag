"""
nilRAG init file.
"""

from .nildb_requests import NilDB, Node  # noqa: F401
from .util import decrypt_float_list  # noqa: F401
from .util import (create_chunks, decrypt_string_list, encrypt_float_list,
                   encrypt_string_list, euclidean_distance,
                   find_closest_chunks, from_fixed_point,
                   generate_embeddings_huggingface, group_shares_by_id,
                   load_file, to_fixed_point)

__version__ = "0.1.0"
