# RAG Example

```shell
pip install -r requirements.txt
```

## Architecture

### FE
1. Create embeddings the chunks
2. Secret share the embeddings and the chunks
3. Send those to NilDB: [store_fe_information](./store.py).

### Client
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
