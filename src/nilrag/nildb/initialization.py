import json
import logging


class NilDBInit:

    async def create_rag_schema(self) -> str:
        """
        Creates a new schema for RAG on all nodes in the cluster concurrently.
        """
        try:
            with open(
                "src/nildb/aux_files/rag_schema.json", "r", encoding="utf8"
            ) as schema_file:
                schema = json.load(schema_file)

            # Create a new schema
            new_schema_id = await self.create_schema(schema, "nilRAG data")
            logging.info("📚 New Schema: %s", new_schema_id)
            logging.info("Store schema in the .env file as SCHEMA_ID.")
            # Updates current schema_id
            self.schema_id = new_schema_id

            return new_schema_id

        except FileNotFoundError:
            logging.error("Schema file not found: src/nildb/aux_files/rag_schema.json")
            raise

        except RuntimeError as error:
            logging.error("Failed to use SecretVaultWrapper: %s", str(error))
            raise

    async def create_clusters_schema(self) -> str:
        """
        Creates a new schema for clustering on all nodes in the cluster concurrently.
        """
        try:
            with open(
                "src/nildb/aux_files/clusters_schema.json", "r", encoding="utf8"
            ) as schema_file:
                schema = json.load(schema_file)

            # Create a new schema
            new_clusters_schema_id = await self.create_schema(
                schema, "Clusters' centroids"
            )
            logging.info("📚 New Schema: %s", new_clusters_schema_id)
            logging.info("Store schema in the .env file as CLUSTERS_SCHEMA_ID.")
            # Updates current schema_id
            self.clusters_schema_id = new_clusters_schema_id

            return new_clusters_schema_id

        except FileNotFoundError:
            logging.error(
                "Schema file not found: src/nildb/aux_files/clusters_schema.json"
            )
            raise

        except RuntimeError as error:
            logging.error("Failed to use SecretVaultWrapper: %s", str(error))
            raise

    async def create_subtract_query(self, subtract_query_id: str = None) -> str:
        """
        Creates a new query on all nodes in the cluster concurrently for the subtraction operation.

        Args:
            subtract_query_id (str, optional): A custom query ID. If not provided, a new UUID is generated.

        Returns:
            str: The created schema id.
        """

        if self.with_clustering == False:
            with open(
                "src/nildb/aux_files/subtract_query.json", "r", encoding="utf8"
            ) as query_file:
                query = json.load(query_file)
                query_name = "Returns the difference between the nilDB embeddings and the query embedding"
        else:
            with open(
                "src/nildb/aux_files/subtract_query_with_clustering.json",
                "r",
                encoding="utf8",
            ) as query_file:
                query = json.load(query_file)
                query_name = "Returns the difference between the nilDB embeddings and the query embedding with a closest centroid tag"

        schema_id = self.schema_id
        subtract_query_id = self.create_query(
            self, query, schema_id, query_name, subtract_query_id
        )
        self.subtract_query_id = subtract_query_id

        return subtract_query_id
