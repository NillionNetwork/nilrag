"""
nilRAG init file.
"""

from .nildb.initialization import NilDBInit
from .nildb.operations import NilDBOps
from .nildb.org_config import ORG_CONFIG
from .utils.benchmark import benchmark_time, benchmark_time_async
from .utils.process import (cluster_embeddings, create_chunks,
                            generate_embeddings_huggingface, load_file)
from .utils.transform import (decrypt_float_list, encrypt_float_list,
                              group_shares_by_id, to_fixed_point)

__all__ = [
    "NilDBInit",
    "NilDBOps",
    "benchmark_time",
    "benchmark_time_async",
    "cluster_embeddings",
    "load_file",
    "create_chunks",
    "generate_embeddings_huggingface",
    "cluster_embeddings",
    "decrypt_float_list",
    "encrypt_float_list",
    "group_shares_by_id",
    "to_fixed_point",
    "ORG_CONFIG",
]

__version__ = "0.1.0"
