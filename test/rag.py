"""
Test suite containing functional unit tests of exported functions.
"""
import unittest
import numpy as np
from src.crypto.secret_sharing import AdditiveSecretSharing
from src.nilrag.util import load_file, create_chunks, generate_embeddings_huggingface, find_closest_chunks


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

    def test_rag_with_secret_sharing(self):
        """
        Test the RAG method with secret sharing.
        """
        secret_sharing = AdditiveSecretSharing(num_parties=3, prime_mod=2**31 - 1, precision=self.precision)

        paragraphs = load_file(self.file_path)
        chunks = create_chunks(paragraphs, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = generate_embeddings_huggingface(chunks)

        # Generate secret shares for embeddings
        all_embedding_shares = [secret_sharing.secret_share(embedding) for embedding in embeddings]
        self.assertEqual(len(all_embedding_shares[0]), secret_sharing.num_parties, "Number of shares per embedding should match the number of parties.")

        # Generate embeddings for the query and create secret shares
        query_embedding = generate_embeddings_huggingface([self.query])[0]
        query_embedding_shares = secret_sharing.secret_share(query_embedding)
        self.assertEqual(len(query_embedding_shares), secret_sharing.num_parties, "Query embedding shares should match the number of parties.")

        # Compute differences for secret shares
        all_differences_shares = [
            [np.array(query_embedding_shares[party]) - np.array(embedding_share[party]) for party in range(secret_sharing.num_parties)]
            for embedding_share in all_embedding_shares
        ]

        # Reveal differences and compute distances
        differences = [secret_sharing.secret_reveal(differences_share) for differences_share in all_differences_shares]
        distances = [np.linalg.norm(diff) for diff in differences]

        # Retrieve top results
        sorted_indices = np.argsort(distances)
        top_k = 2
        top_results = [(chunks[idx], distances[idx]) for idx in sorted_indices[:top_k]]

        # Assertions
        self.check_top_results(top_results)


if __name__ == "__main__":
    unittest.main()
