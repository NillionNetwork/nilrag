# RAG Example

```shell
pip install -r requirements.txt
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


## FE

### 1. Initialization: Setting Up Schemas for nilDB
This initialization step needs to happen before anything else. Note, the
initialization only needs to be run once by the FE.

The FE needs to introduce:
1. `schema`: which is the structure of the data that the FE will store.
    In this case we have `embedding` (`vector<integer>`) and `chunk`
    (`string`). Each FE will upload multiple `embedding`s and `chunk`.

2. `query`: This is the nilDB query that will compute the differences under
    MPC between the stored FE embeddings and the client's embedding.

To initialize the `schema` and `query` call:
```shell
python nilrag/initialize.py
```
This will print a response like:
```shell
Schema ID: c0587a1e-1180-4990-99f8-a7c17a700b80
Query ID: 5a83eb59-71f0-4c8c-8bd7-27330dbec3f1
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
