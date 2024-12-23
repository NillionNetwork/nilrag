# nilRAG [![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/NillionNetwork/nilrag/blob/main/LICENSE)
Retrieval Augmented Generation (RAG) using Nillion's nilDB and nilQL.
RAG is a technique that grants large language models information retrieval capabilities and context that they might be missing.

# Use case

Data owners often possess valuable files that clients wish to query to enhance their LLM-based inferences. However, ensuring privacy is a key challenge: data owners want to keep their data confidential, and clients are equally concerned about safeguarding their queries.

nilRAG addresses this challenge by enabling secure data sharing and querying. It allows data owners to store their data securely in a nilDB cluster while allowing clients to query the data without exposing their queries or compromising the data's privacy.

The process involves leveraging a Trusted Execution Environment (TEE) server for secure computation. Data owners upload their information to the nilDB cluster, while the TEE server processes client queries and retrieves the most relevant results (top-k) without revealing sensitive information from either party.

## Entities summary

Let us deep dive into the entities and their roles in the system.

### Data Owners: Secure stores files for RAG
Data owners contribute multiple files, where each file contains several paragraphs. Before sending the files to the nilDB instances, they are processed into N chunks of data and their corresponding embeddings:

Data Representation:
Chunks (ch_i): Represented as encoded strings.
Embeddings (e_i): Represented as vectors of floats (fixed-point values).

Once the files are encoded into chunks and embeddings, they are blinded before being uploaded to the NilDB, where each chunk and embedding is secret-shared.


### Client: Issues a query q
A client submits a query q to search against the data owners' files stored in NilDB and perform RAG (retrieve the most relevant data and use the top-k results for privacy-preserving machine learning (PPML) inference).

Similar to the data encoding by data owners, the query is processed into its corresponding embeddings:

### NilDB: Secure Storage and Query Handling
NilDB stores the blinded chunks and embeddings provided by data owners. When a client submits a query, NilDB computes the differences between the query’s embeddings and each stored embedding in a privacy-preserving manner:

```python
differences = [embedding - query for embedding in embeddings]
```

Key Points:
- The number of differences (N) corresponds to the number of chunks uploaded by the data owners.
- For secret-sharing-based NilDB, the computation is performed on the shares.

### nilTEE: Secure Processing and Retrieval
The nilTEE performs the following steps:

1. Retrieve and Reveal Differences:
- Connect to NilDB to fetch the blinded differences.
- Reveal the differences by reconstructing shares.

2. Identify Top-k Indices:
- Sort the differences while retaining their indices to find the `top_k` matches:
```python
indexed_diff = list(enumerate(differences))
sorted_indexed_diff = sorted(indexed_diff, key=lambda x: x[1])
indices = [x[0] for x in sorted_indexed_diff]
k = 5
top_k_indices = indices[:k]
```

3. Fetch Relevant Chunks:
- Request NilDB to retrieve the blinded chunks corresponding to the `top_k_indices`.

4. Prepare for Inference:
- Combine the retrieved `top_k_chunks` with the original query.
- Use the data with an LLM inside the nilTEE for secure inference.

# How to use

## Installation
```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate
```

Local installation:
```bash
# Install package in development mode
uv pip install -e .
```

Using pip (not available yet):
```bash
pip install nilrag
```

## Data owner

### Initialization
This initialization step needs to happen before anything else. Note, the
initialization *only needs to be run once* by the data owner.

This initialization introduces:
1. `schema`: which is the structure of the data that the data owner will store.
    In this case we have `embedding` (`vector<integer>`) and `chunk`
    (`string`). Each data owner will upload multiple `embedding`s and `chunk`.
2. `query`: This is the nilDB query that will compute the differences under
    MPC between the stored data owner embeddings and the client's embedding.

In [examples/data_owner_init.py](examples/data_owner_init.py), we provide an example of how to
define the nilDB nodes. Modify [examples/uninitialized_nildb_config.py](examples/uninitialized_nildb_config.py) by adding more nodes and defining the correct URLs, ORGs, and Tokens. 

The nilDB instance is initialized as follows:
```python
nilDB = NilDB(nilDB_nodes)
```
The schema and query are initialized as follows:
```python
# Initialize schema and queries
nilDB.init_schema()
nilDB.init_diff_query()
```

By running the script, the `schema` and `query` are saved to the `initialized_nildb_config.json` file:
```bash
uv run examples/data_owner_init.py
```

### Uploading Documents
After initialization, the data owner can upload their documents to the nilDB instance. We provide an example of how to do this in [examples/data_owner_upload.py](examples/data_owner_upload.py).

By running the script, the documents are uploaded to the nilDB instance in secret-shared form:
```bash
uv run examples/data_owner_upload.py
```

## TEE Server
Start the TEE server with a specific config file:

```bash
# Using default config (tee_nildb_config.json)
uv run launch_tee.py

# Using a custom config file
uv run launch_tee.py -c custom_nildb_config.json
# or
uv run launch_tee.py --config path/to/config.json
```

The server will start on http://0.0.0.0:8000 with:
- API documentation at `/docs`


## Client query
After having nilDB initialized, documents uploaded, and the TEE server running, the client can query the nilDB instance. We provide an example of how to do this in [examples/client_query.py](examples/client_query.py).

By running the script, the client's query is sent to the TEE server and the response is returned:
```bash
uv run examples/client_query.py
```

## Running Tests
```bash
# Run a specific test file
uv run -m unittest test.rag
```

You can also add verbose output with -v:
```bash
uv run -m unittest test.rag -v
```

## Project Structure
```
nilrag/
├── src/
│   └── nilrag/
│       ├── __init__.py          # Package exports
│       ├── __main__.py          
│       ├── app.py               # FastAPI application and TEE server
│       ├── nildb.py             # NilDB and Node classes
│       └── util.py              # Utility functions for RAG
├── test/
│   ├── __init__.py
│   └── rag.py                   # Test suite for RAG functionality
├── examples/
│   ├── client_query.py          # Client query example
│   ├── data_owner_init.py       # Data owner initialization example
│   ├── data_owner_upload.py     # Data owner upload example
│   ├── nildb_config.json        # Example config with initialized nodes
│   ├── tee_nildb_config.json
│   └── uninitialized_nildb_config.json
├── data/
│   └── cities.txt               # Sample data for testing
├── scripts/                     # Utility scripts
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pinned dependencies
├── uv.lock                      # UV lockfile
├── launch_tee.py                # TEE server launcher
└── README.md                    # Documentation
```

### Key Components:
- `src/nilrag/app.py`: TEE server implementation with FastAPI
- `src/nilrag/nildb.py`: Core NilDB interaction logic
- `src/nilrag/util.py`: RAG utilities (embeddings, chunking, rational encoding)
- `test/rag.py`: Test suite for RAG functionality
- `examples`: Script examples for data owner, client, and TEE server
- `data`: Sample data for testing
- `scripts`: Utility scripts examples for maintenance and initialization

