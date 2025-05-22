"""
Performance comparison tests for nilRAG operations.
Compares performance of top_num_chunks_execute between two versions of the code.
"""

import asyncio
import os
import time
import json
import pytest
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from nilrag.nildb.org_config import ORG_CONFIG
from nilrag.rag_vault import RAGVault

# Test configuration
TEST_CONFIG = {
    "tolerance": 0.1,  # 10% tolerance for performance differences
    "warmup_runs": 3,  # Number of warmup runs before actual test
    "test_runs": 5,    # Number of test runs for averaging
    "timeout": 60,     # Maximum time for a single test
    "results_file": "performance_results.json"  # File to store results
}

# Test datasets
TEST_DATASETS = ["1k-fake.txt", "5k-fake.txt", "10k-fake.txt"]
TEST_PROMPT = "Who is Michelle Ross?"
TEST_NUM_CHUNKS_OPTIONS = [1, 2, 5, 10]  # Testing chunk sizes from 1 to 10
TEST_NUM_CLUSTERS = 1  # Fixed number of clusters to search through

def get_current_branch():
    """Get the name of the current git branch."""
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()

def switch_branch(branch_name: str):
    """Switch to the specified git branch."""
    subprocess.run(['git', 'checkout', branch_name], check=True)

@dataclass
class PerformanceResult:
    """Class to store performance test results."""
    dataset: str
    num_chunks: int
    current_time: float
    error_message: str = ""

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "num_chunks": self.num_chunks,
            "current_time": self.current_time,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@pytest.fixture
async def rag_vault():
    """Create and return a RAGVault instance for testing."""
    load_dotenv(override=True)
    
    schema_id = os.getenv("SCHEMA_ID")
    clusters_schema_id = os.getenv("CLUSTERS_SCHEMA_ID")
    subtract_query_id = os.getenv("QUERY_ID")
    
    rag = await RAGVault.create(
        ORG_CONFIG["nodes"],
        ORG_CONFIG["org_credentials"],
        schema_id=schema_id,
        clusters_schema_id=clusters_schema_id,
        subtract_query_id=subtract_query_id,
    )
    return rag

async def run_benchmark(
    rag: RAGVault,
    num_chunks: int,
    num_clusters: int,
    enable_benchmarks: bool = True
) -> float:
    """Run the benchmark and return the execution time."""
    start_time = time.time()
    
    try:
        await asyncio.wait_for(
            rag.top_num_chunks_execute(
                TEST_PROMPT,
                num_chunks,
                enable_benchmarks,
                num_clusters
            ),
            timeout=TEST_CONFIG["timeout"]
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Benchmark exceeded timeout of {TEST_CONFIG['timeout']} seconds")
    
    end_time = time.time()
    return end_time - start_time

async def run_performance_test(
    rag: RAGVault,
    dataset: str,
    num_chunks: int,
    num_clusters: int
) -> PerformanceResult:
    """Run a complete performance test with warmup and multiple runs."""
    # Warmup runs
    for _ in range(TEST_CONFIG["warmup_runs"]):
        await run_benchmark(rag, num_chunks, num_clusters)
    
    # Actual test runs
    times = []
    
    for _ in range(TEST_CONFIG["test_runs"]):
        time_taken = await run_benchmark(rag, num_chunks, num_clusters)
        times.append(time_taken)
    
    # Calculate average
    avg_time = sum(times) / len(times)
    
    return PerformanceResult(
        dataset=dataset,
        num_chunks=num_chunks,
        current_time=avg_time
    )

def save_results(results: List[PerformanceResult], branch_name: str):
    """Save test results to a file."""
    # Load existing results if any
    all_results = {}
    if os.path.exists(TEST_CONFIG["results_file"]):
        with open(TEST_CONFIG["results_file"], 'r') as f:
            all_results = json.load(f)
    
    # Add new results
    all_results[branch_name] = [r.to_dict() for r in results]
    
    # Save all results
    with open(TEST_CONFIG["results_file"], 'w') as f:
        json.dump(all_results, f, indent=2)

def compare_results(main_results: List[PerformanceResult], feature_results: List[PerformanceResult]):
    """Compare results between main and feature branches."""
    print("\nPerformance Comparison Results:")
    
    for main, feature in zip(main_results, feature_results):
        time_difference = (feature.current_time - main.current_time) / main.current_time
        
        print(f"\nDataset: {main.dataset}")
        print(f"Configuration: {main.num_chunks} chunks")
        print(f"Main Branch Time: {main.current_time:.2f}s")
        print(f"Feature Branch Time: {feature.current_time:.2f}s")
        print(f"Difference: {time_difference:.2%}")
        
        if time_difference > TEST_CONFIG["tolerance"]:
            print(f"WARNING: Performance regression detected!")
            print(f"Feature branch is {time_difference:.2%} slower than main branch")

@pytest.mark.parametrize("dataset", TEST_DATASETS)
@pytest.mark.parametrize("num_chunks", TEST_NUM_CHUNKS_OPTIONS)
def test_top_num_chunks_performance(rag_vault, event_loop, dataset, num_chunks):
    """Test performance of top_num_chunks_execute and save results."""
    try:
        result = event_loop.run_until_complete(
            run_performance_test(rag_vault, dataset, num_chunks, TEST_NUM_CLUSTERS)
        )
        
        # Get current branch name
        branch_name = get_current_branch()
        
        # Save result
        save_results([result], branch_name)
        
        # Print current result
        print(f"\nTest Results for {branch_name}:")
        print(f"Dataset: {dataset}")
        print(f"Configuration: {num_chunks} chunks")
        print(f"Execution Time: {result.current_time:.2f}s")
        
    except TimeoutError as e:
        pytest.fail(f"Test timed out: {str(e)}")
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

def run_tests_for_branch(branch_name: str):
    """Run tests for a specific branch."""
    switch_branch(branch_name)
    pytest.main(['test/test_performance.py', '-v', '-s'])

def run_all_tests():
    """Run tests on both main and feature branches."""
    # Store current branch
    current_branch = get_current_branch()
    
    try:
        # Run tests on test_1 branch (main)
        print("\nRunning tests on test_1 branch...")
        switch_branch('test_1')
        pytest.main(['test/test_performance.py', '-v', '-s'])
        
        # Switch back to test_2 branch
        print("\nSwitching back to test_2 branch...")
        switch_branch('test_2')
        
        # Run tests on test_2 branch
        print("\nRunning tests on test_2 branch...")
        pytest.main(['test/test_performance.py', '-v', '-s'])
        
        # Compare results
        compare_branches()
        
    finally:
        # Always switch back to original branch
        switch_branch(current_branch)

def compare_branches():
    """Compare results between main and feature branches."""
    if not os.path.exists(TEST_CONFIG["results_file"]):
        print("No results file found. Run tests on both branches first.")
        return
    
    with open(TEST_CONFIG["results_file"], 'r') as f:
        all_results = json.load(f)
    
    if 'test_1' not in all_results or len(all_results) < 2:
        print("Need results from both test_1 and test_2 branches to compare.")
        return
    
    # Get results from both branches
    main_results = [PerformanceResult.from_dict(r) for r in all_results['test_1']]
    feature_results = [PerformanceResult.from_dict(r) for r in all_results['test_2']]
    
    # Compare results
    compare_results(main_results, feature_results)

if __name__ == "__main__":
    run_all_tests() 