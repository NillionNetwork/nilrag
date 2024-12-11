# nilRAG [![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/NillionNetwork/nilrag/blob/main/LICENSE)
Retrieval Augmented Generation (RAG) using Nillion's nilDB and nilQL.
RAG is a technique that grants large language models information retrieval capabilities and context that they might be missing.


## Set Up & Tests
Install the required packages:
```shell
pip install -r requirements.txt
```

Test RAG locally:
```shell
python -m unittest test.rag
```

# Architecture

```shell
. nilrag/
|-- initialization/
|   |-- initialize.py
|-- fe/
|   |-- upload_data.py
|-- client/
|   |-- upload_query.py
```


## FE (i.e., Data Owner)

### 1. Initialization (i.e., Setting Up Schemas for nilDB)
This initialization step needs to happen before anything else. Note, the
initialization *only needs to be run once* by the FE.

This initialization introduces:
1. `schema`: which is the structure of the data that the FE will store.
    In this case we have `embedding` (`vector<integer>`) and `chunk`
    (`string`). Each FE will upload multiple `embedding`s and `chunk`.
2. `query`: This is the nilDB query that will compute the differences under
    MPC between the stored FE embeddings and the client's embedding.

In [src/nilrag/nildb.py](src/nilrag/nildb.py), we provide an example of how to
define the nilDB nodes. Modify this by adding more nodes and defining the
correct URLs, ORGs, and Tokens:
```python
nilDB_nodes = [
    Node(
        url="https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1/",
        org="b3d3f64d-ef12-41b7-9ff1-0e7681947bea",
        bearer_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0",
    ),

    # Add more nodes here...
]
nilDB = NilDB(nilDB_nodes)
```

After more nodes have been added, initialize the `schema` and `query` by
calling:
```shell
$ python src/nilrag/nildb.py                                                                                                                            [17:15:41]

    No configuration file found. Initializing NilDB...
    NilDB configuration saved to file.
    NilDB instance: URL: https://nil-db.sandbox.app-cluster.sandbox.nilogy.xyz/api/v1
    Org: b3d3f64d-ef12-41b7-9ff1-0e7681947bea
    Bearer Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2QzZjY0ZC1lZjEyLTQxYjctOWZmMS0wZTc2ODE5NDdiZWEiLCJ0eXBlIjoiYWNjZXNzLXRva2VuIiwiaWF0IjoxNzMyODkzMzkwfQ.x62bCqtz6mwYhz9ZKXYuD2EIu073fxmPKCh6UkWyox0
    Schema ID: 5973c7cc-cfcf-49fb-af4a-0e3db89fcbca
    Query ID: f45d608a-26a9-4c9f-927f-a63505812db2
```

### 2. FE Uploads Documents
The FE splits their documents into `Embeddings` and `Chunks`, which are both
vectors of the same size.


1. Create embeddings the chunks
2. Secret share the embeddings and the chunks
3. Send those to NilDB: [store_fe_information](./store.py).

## Client
1. Create embeddings of the query
2. Send the query to the Server (TEE)
3. Wait for response

### Server (TEE) -- Driver
1. Receive query from client
2. Secret share query and send to NilDB
3. Ask NilDB to compute the differences
4. Wait for response
5. Ask NilDB to return top k chunks
6. Run LLM.
7. Return answer to the user.

### NilDB
1. Receive Query
2. Compute Differences
3. Respond to Server
