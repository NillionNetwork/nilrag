"""
Test suite containing functional unit tests of exported functions.
"""

import json
import os
import time
import unittest
from dataclasses import dataclass
from typing import List, Tuple

import nilql
import numpy as np
from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault, find_closest_chunks
from nilrag.utils.process import (create_chunks,
                                  generate_embeddings_huggingface, load_file)
from nilrag.utils.transform import decrypt_float_list, encrypt_float_list

DEFAULT_PROMPT = "Who is Michelle Ross?"
RUN_OPTIONAL_TESTS = False


@dataclass
class TestCase:
    """Test case configuration for RAG testing."""

    file_path: str
    query: str
    expected_results: List[Tuple[str, float]]


class TestRAGMethods(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for RAG (Retrieval Augmented Generation) functionality.

    Tests both plaintext and secret-shared implementations of RAG, including
    chunk creation, embedding generation, similarity search, and result retrieval.
    """

    def setUp(self):
        """
        Setup shared resources for tests.
        """
        self.chunk_size = 50
        self.overlap = 10
        self.precision = 7

        # Expected top results and queries
        self.test_cases = [
            TestCase(
                file_path="examples/data/cities.txt",
                query="Tell me about places in Asia.",
                expected_results=[
                    (
                        "Tokyo, Japan's bustling capital, is known for its modern "
                        "architecture, cherry blossoms, and incredible food scene. "
                        "The Shinjuku and Shibuya areas are hotspots for tourists. "
                        "Other must-visit sites include the historic Asakusa district, "
                        "the Meiji Shrine, and Akihabara, a hub for tech and anime "
                        "enthusiasts.",
                        1.04,
                    ),
                    (
                        "Kyoto, once the capital of Japan, is famous for its "
                        "classical Buddhist temples, as well as gardens, imperial "
                        "palaces, Shinto shrines, and traditional wooden houses. "
                        "The city is also known for its formal traditions such as "
                        "kaiseki dining and geisha female entertainers.",
                        1.08,
                    ),
                ],
            ),
            TestCase(
                file_path="examples/data/computer-science.txt",
                query="Tell me about encryption.",
                expected_results=[
                    (
                        "Cryptography: Cryptography secures communication through "
                        "encryption and decryption. It is essential for data privacy "
                        "and authentication. Modern cryptographic algorithms include "
                        "AES, RSA, and ECC.",
                        0.79,
                    ),
                    (
                        "Cybersecurity: Cybersecurity involves protecting computer "
                        "systems and networks from unauthorized access or attacks. "
                        "Techniques include encryption, firewalls, and intrusion "
                        "detection systems. It is crucial in safeguarding sensitive "
                        "data.",
                        1.02,
                    ),
                ],
            ),
            TestCase(
                file_path="examples/data/climate-change.txt",
                query="Tell me about global warming.",
                expected_results=[
                    (
                        "Climate change is one of the most pressing challenges of "
                        "our time, driven primarily by greenhouse gas emissions from "
                        "human activities. The effects include rising global "
                        "temperatures, melting ice caps, and more frequent extreme "
                        "weather events. Solutions require a combination of renewable "
                        "energy adoption, reforestation, and policy changes.",
                        0.90,
                    ),
                    (
                        "Climate change is one of the most pressing challenges of "
                        "our time, driven primarily by greenhouse gas emissions from "
                        "human activities. The effects include rising global "
                        "temperatures, melting ice caps, and more frequent extreme "
                        "weather events. Solutions require a combination of renewable "
                        "energy adoption, reforestation, and policy changes.",
                        0.90,
                    ),
                ],
            ),
        ]

    def check_top_results(self, top_results, expected_results):
        """
        Helper function to verify that the top results match the expected output.
        """
        self.assertEqual(
            len(top_results),
            len(expected_results),
            "Number of top results should match expected results.",
        )
        for (result_chunk, result_distance), (expected_chunk, expected_distance) in zip(
            top_results, expected_results
        ):
            self.assertEqual(
                result_chunk, expected_chunk, f"Chunk mismatch: {result_chunk}"
            )
            self.assertAlmostEqual(
                result_distance,
                expected_distance,
                delta=0.01,
                msg=f"Distance mismatch for chunk: {result_chunk}",
            )

    def test_rag_plaintext(self):
        """
        Test the plaintext RAG method for retrieving top results.
        """
        for case in self.test_cases:
            paragraphs = load_file(case.file_path)
            chunks = create_chunks(
                paragraphs, chunk_size=self.chunk_size, overlap=self.overlap
            )
            embeddings = generate_embeddings_huggingface(chunks)

            # Generate embeddings for the query
            query_embedding = generate_embeddings_huggingface([case.query])[0]

            # Find closest chunks
            top_results = find_closest_chunks(query_embedding, chunks, embeddings)

            # Assertions
            self.check_top_results(top_results, case.expected_results)

    def _prepare_encrypted_data(self, paragraphs, query, num_parties):
        """Prepare encrypted data for RAG testing."""
        chunks = create_chunks(
            paragraphs, chunk_size=self.chunk_size, overlap=self.overlap
        )
        embeddings = generate_embeddings_huggingface(chunks)
        query_embedding = generate_embeddings_huggingface([query])[0]

        # Generate encryption keys
        additive_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"sum": True}
        )
        xor_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"store": True}
        )

        # Encrypt data
        nilql_embeddings = [
            encrypt_float_list(additive_key, embedding) for embedding in embeddings
        ]
        nilql_chunks = [nilql.encrypt(xor_key, chunk) for chunk in chunks]
        nilql_query_embeddings = encrypt_float_list(additive_key, query_embedding)

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "nilql_chunks": nilql_chunks,
            "nilql_embeddings": nilql_embeddings,
            "nilql_query_embeddings": nilql_query_embeddings,
            "additive_key": additive_key,
            "xor_key": xor_key,
        }

    def _compute_differences(self, encrypted_data, num_parties):
        """Compute differences between query and document embeddings."""
        # Rearrange shares by party
        embeddings_shares = [
            [
                [row[party] for row in embedding]
                for embedding in encrypted_data["nilql_embeddings"]
            ]
            for party in range(num_parties)
        ]
        query_embedding_shares = [
            [entry[party] for entry in encrypted_data["nilql_query_embeddings"]]
            for party in range(num_parties)
        ]

        # Compute differences
        differences_shares = [
            np.array(query_embedding_shares[party]) - np.array(embeddings_shares[party])
            for party in range(num_parties)
        ]

        # Restructure differences for nilQL
        nilql_differences = [
            [
                [int(differences_shares[party][i][j]) for party in range(num_parties)]
                for j in range(len(differences_shares[0][i]))
            ]
            for i in range(len(differences_shares[0]))
        ]

        return nilql_differences

    def test_rag_with_nilql(self):
        """
        Test the RAG method with secret sharing.
        """
        num_parties = 2
        top_k = 2

        for case in self.test_cases:
            start_time = time.time()
            paragraphs = load_file(case.file_path)
            num_paragraphs = len(paragraphs)

            # Prepare encrypted data
            encrypted_data = self._prepare_encrypted_data(
                paragraphs, case.query, num_parties
            )

            # Compute differences
            nilql_differences = self._compute_differences(encrypted_data, num_parties)

            # Reveal differences and compute distances
            differences = [
                decrypt_float_list(encrypted_data["additive_key"], differences_share)
                for differences_share in nilql_differences
            ]
            distances = [np.linalg.norm(diff) for diff in differences]

            # Get top results
            sorted_indices = np.argsort(distances)
            nilql_top_results = [
                encrypted_data["nilql_chunks"][idx] for idx in sorted_indices[:top_k]
            ]

            # Reveal chunks and pair with distances
            top_results = [
                (
                    nilql.decrypt(encrypted_data["xor_key"], chunk),
                    distances[idx],
                )
                for chunk, idx in zip(nilql_top_results, sorted_indices[:top_k])
            ]

            elapsed_time = time.time() - start_time
            print(
                f"Test case '{case.query}' completed in {elapsed_time:.2f} seconds. "
                f"({num_paragraphs} total paragraphs)"
            )

            # Assertions
            self.check_top_results(top_results, case.expected_results)

    @unittest.skipUnless(RUN_OPTIONAL_TESTS, "Skipping optional test.")
    async def test_top_num_chunks_execute(self):
        """
        Test the RAG method with nilDB.
        """

        # Load environment variables
        load_dotenv(override=True)

        schema_id = os.getenv("SCHEMA_ID")
        clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
        subtract_query_id = os.getenv("QUERY_ID")

        # Initialize vault with clustering enabled
        rag = await RAGVault.create(
            ORG_CONFIG["nodes"],
            ORG_CONFIG["org_credentials"],
            schema_id=schema_id,
            clusters_schema_id=clusters_schema_id,
            subtract_query_id=subtract_query_id,
        )

        print("Perform nilRAG...")
        start_time = time.time()
        query = DEFAULT_PROMPT
        top_chunks = await rag.top_num_chunks_execute(query, 2)
        end_time = time.time()
        print(json.dumps(top_chunks, indent=4))
        print(f"Query took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    unittest.main()
