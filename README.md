# nilRAG [![PyPI version and link](https://badge.fury.io/py/nilrag.svg)](https://badge.fury.io/py/nilrag) [![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/NillionNetwork/nilrag/blob/main/LICENSE)

Retrieval Augmented Generation (RAG) using Nillion's
[nilAI](https://github.com/NillionNetwork/nilAI),
[nilDB](https://github.com/NillionNetwork/nildb), and
[nilQL](https://github.com/NillionNetwork/nilql-py). RAG is a technique that
grants large language models information retrieval capabilities and context that
they might be missing.

nilRAG combines multiparty computation (MPC) and trusted execution environment
(TEE) technologies.

# Overview

Data owners often possess valuable files that clients wish to query to enhance
their LLM-based inferences. However, ensuring privacy is a key challenge: data
owners want to keep their data confidential, and clients are equally concerned
about safeguarding their queries.

nilRAG addresses this challenge by enabling secure data sharing and querying. It
allows data owners to store their data securely in a nilDB cluster while
allowing clients to query the data without exposing their queries or
compromising the data's privacy.

The process involves leveraging a Trusted Execution Environment (TEE) server for
secure computation through nilAI. Data owners upload their information to the
nilDB cluster, while nilAI processes client queries and retrieves the most
relevant results (top-k) without revealing sensitive information from either
party.

## Entities summary

Let us deep dive into the entities and their roles in the system.

1) **Data Owners:** Secure stores files for RAG Data owners contribute multiple
files, where each file contains several paragraphs. Before sending the files to
the nilDB instances, they are processed into N chunks of data and their
corresponding embeddings:
    ```
    Chunks (ch_i): Represented as encoded strings.
    Embeddings (e_i): Represented as vectors of floats (fixed-point values).
    ```

    Once the files are encoded into chunks and embeddings, they are blinded before being uploaded to the NilDB, where each chunk and embedding is secret-shared.

2) **Client:** Issues a query q A client submits a query q to search against the
data owners' files stored in NilDB and perform RAG (retrieve the most relevant
data and use the top-k results for privacy-preserving machine learning (PPML)
inference).

    Similar to the data encoding by data owners, the query is processed into its corresponding embeddings:

3) **NilDB:** Secure Storage and Query Handling
    NilDB stores the blinded chunks and embeddings provided by data owners. When a client submits a query, NilDB computes the differences between the query's embeddings and each stored embedding in a privacy-preserving manner:
    ```python
    differences = [embedding - query for embedding in embeddings]
    ```

    Key Points:
    - The number of differences (N) corresponds to the number of chunks uploaded by the data owners.
    - For secret-sharing-based NilDB, the computation is performed on the shares.

4) **nilAI:** Secure Processing and Retrieval The nilTEE performs the following
steps:
    1. Retrieve and Reveal Differences: Connect to NilDB to fetch the blinded
       differences and then reveal the differences by reconstructing shares.

    2. Identify Top-k Indices: Sort the differences while retaining their
       indices to find the `top_k` matches:
        ```python
        indexed_diff = list(enumerate(differences))
        sorted_indexed_diff = sorted(indexed_diff, key=lambda x: x[1])
        indices = [x[0] for x in sorted_indexed_diff]
        k = 5
        top_k_indices = indices[:k]
        ```

    3. Fetch Relevant Chunks: Request NilDB to retrieve the blinded chunks
       corresponding to the `top_k_indices`.

    4. Prepare for Inference: Combine the retrieved `top_k_chunks` with the
       original query. Use the data with an LLM inside the nilTEE for secure
       inference.

# How to use

## Installation
First install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:
```shell
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate
```

Then either follow the local installation:
```shell
# Install package in development mode
uv pip install -e .
```
or use `pip`:
```shell
pip install nilrag
```

## Data owner

### 1. Initialization
First, copy the sample nilDB config file:
```shell
cp ./examples/nildb_config.sample.json ./examples/nildb_config.json
```
Next, register a new organization in Nillion's [SecretVault Registration
Portal](https://sv-sda-registration.replit.app/) and fill in the details in your
nilDB config file `./examples/nildb_config.json`.

You can safely ignore `bearer_token`, `schema_id`, and `diff_query_id` as we'll
fill these out later.

You are all set to create your first schema and query for RAG. At the minimum,
they should look like:
1. `schema`: which is the structure of the data that the data owner will store.
    In this case we have `embedding` (`vector<integer>`) and `chunk` (`string`).
    Each data owner will upload multiple `embedding`s and `chunk`.
2. `query`: This is the nilDB query that will compute the differences under
    MPC between the stored data owner embeddings and the client's embedding.

We have an example that creates a schema and a query, run it as:
```shell
# Use default config file
uv run examples/1.init_schema_query.py

# Or specify a custom config file
uv run examples/1.init_schema_query.py --config /path/to/your/config.json
```
This, will fill out `bearer_token`, `schema_id`, and `diff_query_id` in your
config file. Verify that it has been populated successfully.


### 2. Uploading Documents
After initialization, the data owner can upload their documents to the nilDB
instance. We provide an example of how to do this in
[examples/2.data_owner_upload.py](examples/2.data_owner_upload.py).

By running the script, the documents are uploaded to the nilDB instance in secret-shared form:
```shell
# Use default config and data file
uv run examples/2.data_owner_upload.py

# Or specify custom config and data files
uv run examples/2.data_owner_upload.py --config /path/to/config.json --file /path/to/data.txt
```

## 3. Client Query
After having nilDB initialized, documents uploaded, and access to nilAI, the
client can query the nilDB instance. We provide an example of how to do this in
[examples/3.client_query.py](examples/3.client_query.py).

By running the script, the client's query is sent to nilAI and the response is
returned:
```shell
# Use default config and prompt
uv run examples/3.client_query.py

# Or specify custom config and prompt
uv run examples/3.client_query.py --config /path/to/config.json --prompt "Your custom query here"
```

## Running Tests
```shell
# Run a specific test file
uv run -m unittest test.rag
```
