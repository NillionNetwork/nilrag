# pylint: disable=no-member
"""
This module provides the operations functions for nilDB, including data writing and query
execution.
"""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List
from uuid import uuid4

from nilrag.utils.process import check_inputs_to_upload


class NilDBOps:
    """
    NilDBOps is a class that provides the operations functions for nilDB, including data
    writing and query execution.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    async def write_rag_data(
        self,
        lst_embedding_shares: list[list[int]],
        lst_chunk_shares: list[list[bytes]],
        batch_size: int = 100,
        labels: list[int] | None = None,
        centroids: list[int] | None = None,
    ) -> None:
        """
        Write embeddings and chunks to all nilDB nodes asynchronously in batches.

        Args:
            lst_embedding_shares (list): List of embedding shares for each document,
                e.g. for 3 nodes:
                [
                    # First document's embedding vector (384 dimensions)
                    [
                        # First dimension split into 3 shares (sum mod 2^32)
                        [1234567890, 987654321, 2072745085],
                        # Second dimension split into 3 shares (sum mod 2^32)
                        [3141592653, 2718281828, 3435092815],
                        # ... 382 more dimensions, each split into 3 shares
                    ],
                    # More documents...
                ]
            lst_chunk_shares (list): List of chunk shares for each document, e.g. for 3 nodes:
                [
                    [  # First document's chunk shares
                        b"encrypted chunk for node 1",
                        b"encrypted chunk for node 2",
                        b"encrypted chunk for node 3"
                    ],
                    # More documents...
                ]
            batch_size (int, optional): Number of documents to upload in each batch.
                Defaults to 100.
            labels (list[int]): List of labels given for each embedding.
                Defaults to None.
            centroids (list[int]): List of clusters centroids when clustering is performed.
                Defaults to None.

        Raises:
            AssertionError: If number of embeddings and chunks don't match
            ValueError: If write fails on any nilDB node
        """

        # Input sanity check
        check_inputs_to_upload(
            lst_embedding_shares, lst_chunk_shares, labels, centroids
        )

        # pylint: disable=too-many-locals
        async def process_batch(batch_start: int, batch_end: int) -> None:
            """Process and write a single batch of documents."""
            print(
                f"Processing batch {batch_start//batch_size + 1}: "
                f"documents {batch_start} to {batch_end}"
            )
            # Generate document IDs for this batch
            doc_ids = [str(uuid4()) for _ in range(batch_start, batch_end)]
            tasks = []
            for node_idx, node in enumerate(self.nodes):
                batch_data = []
                for batch_idx, doc_idx in enumerate(range(batch_start, batch_end)):
                    # Join the shares of one embedding in one vector for this node
                    batch_entry = {
                        "_id": doc_ids[batch_idx],
                        "embedding": [
                            e[node_idx] for e in lst_embedding_shares[doc_idx]
                        ],
                        "chunk": lst_chunk_shares[doc_idx][node_idx],
                    }
                    # In case clustering is performed,
                    # join the clusters centroid of the corresponding embedding
                    if (
                        labels is not None
                        and centroids is not None
                        and len(centroids) > 1
                    ):
                        batch_entry["cluster_centroid"] = int(labels[doc_idx])
                    # Add this entry to the batch data
                    batch_data.append(batch_entry)
                rag_schema_id = self.schema_id
                tasks.append(self.write_data_to_node(node, batch_data, rag_schema_id))
            try:
                results = await asyncio.gather(*tasks)
                print(f"Successfully uploaded batch {batch_start//batch_size + 1}")
                for result in results:
                    print(
                        {
                            "status_code": 200,
                            "message": "Success",
                            "response_json": result,
                        }
                    )
            except Exception as e:
                print(f"Error uploading batch {batch_start//batch_size + 1}: {str(e)}")
                raise

        # Process data in batches
        total_documents = len(lst_embedding_shares)
        for batch_start in range(0, total_documents, batch_size):
            batch_end = min(batch_start + batch_size, total_documents)
            await process_batch(batch_start, batch_end)
        # After processing all batches, upload centroids if they exist
        if centroids is not None and len(centroids) > 1:
            await self.write_centroids_data(centroids)

    async def write_centroids_data(self, centroids: List[int]) -> None:
        """
        Writes the centroids data to all nodes after generating the appropriate IDs.
        """
        # Generate IDs for centroids
        centroid_ids = [str(uuid4()) for _ in centroids]

        # Collect tasks for uploading centroids
        tasks = []
        clusters_schema_id = self.clusters_schema_id
        for _, node in enumerate(self.nodes):
            centroids_data = [
                {"_id": centroid_ids[centroid_idx], "cluster_centroid": centroid}
                for centroid_idx, centroid in enumerate(centroids)
            ]
            tasks.append(
                self.write_data_to_node(node, centroids_data, clusters_schema_id)
            )

        # Gather the results of all upload tasks
        try:
            results = await asyncio.gather(*tasks)
            print("Successfully uploaded centroids")
            for result in results:
                print(
                    {
                        "status_code": 200,
                        "message": "Success",
                        "response_json": result,
                    }
                )
        except Exception as e:
            print(f"Error uploading centroids: {str(e)}")
            raise

    async def write_data_to_node(
        self, node: Dict[str, str], node_data: List[Dict], schema_id: str
    ) -> Dict:
        """Write a batch of data to a specific node."""
        try:
            jwt_token = await self.generate_node_token(node["did"])
            payload = {
                "schema": schema_id,
                "data": node_data,
            }
            result = await self.make_request(
                node["url"],
                "data/create",
                jwt_token,
                payload,
            )
            return {"node": node["url"], "result": result}
        except RuntimeError as e:
            print(f"❌ Failed to write to {node['url']}: {str(e)}")
            return {"node": node["url"], "error": str(e)}

    async def execute_subtract_query(
        self,
        nilql_query_embedding: List[List[bytes]],
        closest_centroids: List[int] | None = None,
    ) -> Dict[str, List[int]]:
        """
        Execute the subtract query across all nilDB nodes asynchronously.

        Args:
            nilql_query_embedding (list): Encrypted query embedding for all nilDB nodes.
            closest_centroids (list, optional): The closest centroids to filter by.

        Returns:
            list: List of subtracted shares from each nilDB node.

        Raises:
            ValueError: If query execution fails on any nilDB node.
        """
        # Rearrange nilql_query_embedding to group by party
        query_embedding_shares = [
            [entry[party] for entry in nilql_query_embedding]
            for party in range(len(self.nodes))
        ]

        # Execute queries on all nodes in parallel
        tasks = []
        closest_centroids_key_value = (
            {"closest_centroids": closest_centroids}
            if closest_centroids is not None
            else {}
        )
        for node_index, node in enumerate(self.nodes):
            payload = {
                "id": self.subtract_query_id,
                "variables": {
                    "query_embedding": query_embedding_shares[node_index],
                    **closest_centroids_key_value,
                },
            }
            task = self.execute_query_on_single_node(node, payload)
            tasks.append(task)

        results_from_all_nodes = await asyncio.gather(*tasks)

        # Groups records from different nodes by _id field
        shares_by_id = defaultdict(list)
        for node_result in results_from_all_nodes:
            for share in node_result.get("data", []):
                shares_by_id[share["_id"]].append(share["difference"])
        return shares_by_id

    async def read_chunk_from_nodes(self, chunk_ids: List[str]) -> List:
        """
        Read chunks from all nilDB nodes asynchronously.
        """
        data_filter = {"_id": {"$in": chunk_ids}}
        chunk_shares = await self.read_from_nodes(data_filter, unify_data=False)

        return chunk_shares

    async def read_from_node(
        self, node: Dict[str, str], schema_id: str, data_filter: Dict[str, Any] = None
    ):
        """
        Reads data from a single node and returns the data.

        Args:
            node (Dict[str, str]): The target node to read from.
            data_filter (Dict[str, Any], optional): The filter to apply when reading data.
            schema_id (str): The schema ID to read from.

        Returns:
            Dict[str, Any]: A dictionary containing the response from the single node.
        """
        try:
            jwt_token = await self.generate_node_token(node["did"])
            payload = {
                "schema": schema_id,
                "filter": data_filter or {},
            }
            result = await self.make_request(
                node["url"],
                "data/read",
                jwt_token,
                payload,
            )
            return result.get("data", [])
        except RuntimeError as e:
            print(f"❌ Failed to read from {node['url']}: {str(e)}")
            return {"node": node["url"], "error": str(e)}
