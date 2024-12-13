"""
Test suite containing functional unit tests of exported functions.
"""
import unittest
import numpy as np
import nilql
from src.nilrag.util import load_file, create_chunks, generate_embeddings_huggingface, find_closest_chunks, encrypt_string_list, encrypt_float_list, decrypt_float_list


class TestRAGMethods(unittest.TestCase):
    def setUp(self):
        """
        Setup shared resources for tests.
        """
        self.file_path = "data/cities.txt"
        self.chunk_size = 50
        self.overlap = 10
        self.query = "Tell me about places in Asia."
        self.precision = 7

        # Expected top results
        self.expected_results = [
            ("Tokyo, Japan's bustling capital, is known for its modern architecture, cherry blossoms, and incredible food scene. "
             "The Shinjuku and Shibuya areas are hotspots for tourists. Other must-visit sites include the historic Asakusa district, "
             "the Meiji Shrine, and Akihabara, a hub for tech and anime enthusiasts.", 1.04),
            ("Kyoto, once the capital of Japan, is famous for its classical Buddhist temples, as well as gardens, imperial palaces, "
             "Shinto shrines, and traditional wooden houses. The city is also known for its formal traditions such as kaiseki dining "
             "and geisha female entertainers.", 1.08),
        ]

    def check_top_results(self, top_results):
        """
        Helper function to verify that the top results match the expected output.
        """
        self.assertEqual(len(top_results), len(self.expected_results), "Number of top results should match expected results.")
        for (result_chunk, result_distance), (expected_chunk, expected_distance) in zip(top_results, self.expected_results):
            self.assertEqual(result_chunk, expected_chunk, f"Chunk mismatch: {result_chunk}")
            self.assertAlmostEqual(result_distance, expected_distance, delta=0.01, msg=f"Distance mismatch for chunk: {result_chunk}")

    def test_rag_plaintext(self):
        """
        Test the plaintext RAG method for retrieving top results.
        """
        paragraphs = load_file(self.file_path)
        chunks = create_chunks(paragraphs, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = generate_embeddings_huggingface(chunks)

        # Generate embeddings for the query
        query_embedding = generate_embeddings_huggingface([self.query])[0]

        # Find closest chunks
        top_results = find_closest_chunks(query_embedding, chunks, embeddings)

        # Assertions
        self.check_top_results(top_results)

    def test_rag_with_nilql(self):
        """
        Test the RAG method with secret sharing.
        """
        num_parties = 2  # Replace with dynamic value if needed
        additive_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'sum': True})
        xor_key = nilql.secret_key({'nodes': [{}] * num_parties}, {'store': True})

        # Load paragraphs and create chunks
        paragraphs = load_file(self.file_path)
        chunks = create_chunks(paragraphs, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = generate_embeddings_huggingface(chunks)

        # Encrypt embeddings to generate secret shares with nilQL
        nilql_embeddings = [encrypt_float_list(additive_key, embedding) for embedding in embeddings]
        nilql_chunks = [nilql.encrypt(xor_key, chunk) for chunk in chunks]

        # Rearrange all_embedding_shares to group by party
        embeddings_shares = [
            [[row[party] for row in embedding] for embedding in nilql_embeddings]
            for party in range(num_parties)
        ]
        chunks_shares = [
            [chunk[party] for chunk in nilql_chunks]
            for party in range(num_parties)
        ]

        # Ensure the number of shares matches the number of parties
        self.assertEqual(len(embeddings_shares), num_parties, "Number of shares per embedding should match the number of parties.")
        self.assertEqual(len(chunks_shares), num_parties, "Number of shares per chunk should match the number of parties.")
        self.assertEqual(len(chunks_shares[0]), len(embeddings_shares[0]), "Number of chunks should match the number of embeddings.")

        # Generate embeddings for the query and create secret shares
        query_embedding = generate_embeddings_huggingface([self.query])[0]
        nilql_query_embeddings = encrypt_float_list(additive_key, query_embedding)
        # Rearrange query_embedding_shares to group by party
        query_embedding_shares = [
            [entry[party] for entry in nilql_query_embeddings]
            for party in range(num_parties)
        ]
        self.assertEqual(len(query_embedding_shares), num_parties, "Query embedding shares should match the number of parties.")

        # Compute differences for secret shares. This would happen in nilDB.
        differences_shares = [
            np.array(query_embedding_shares[party]) - np.array(embeddings_shares[party])
            for party in range(num_parties)
        ]
        self.assertEqual(len(differences_shares), num_parties, "Differences embedding shares should match the number of parties.")

        # Restructure the array of differences so it can be revealed next by nilQL.
        nilql_differences = [
            [
                [differences_shares[party][i][j] for party in range(num_parties)]
                for j in range(len(differences_shares[0][i]))
            ]
            for i in range(len(differences_shares[0]))
        ]

        # Reveal differences and compute distances.
        differences = [
            decrypt_float_list(additive_key, differences_share)
            for differences_share in nilql_differences
        ]
        distances = [np.linalg.norm(diff) for diff in differences]

        # Retrieve top results
        sorted_indices = np.argsort(distances)
        top_k = 2
        nilql_top_results = [nilql_chunks[idx] for idx in sorted_indices[:top_k]]

        # Reveal chunks and pair them with their corresponding distances
        top_results = [
            (nilql.decrypt(xor_key, chunk), distances[idx])
            for chunk, idx in zip(nilql_top_results, sorted_indices[:top_k])
        ]

        # Assertions
        self.check_top_results(top_results)

if __name__ == "__main__":
    unittest.main()
