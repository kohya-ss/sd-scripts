"""
Benchmark script to measure performance improvement from caching shapes in memory.

Simulates the get_shape() calls that happen during training.
"""

import time
import tempfile
import torch
from pathlib import Path
from library.cdc_fm import CDCPreprocessor, GammaBDataset


def create_test_cache(num_samples=500, shape=(16, 64, 64)):
    """Create a test CDC cache file"""
    preprocessor = CDCPreprocessor(
        k_neighbors=16, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
    )

    print(f"Creating test cache with {num_samples} samples...")
    for i in range(num_samples):
        latent = torch.randn(*shape, dtype=torch.float32)
        preprocessor.add_latent(latent=latent, global_idx=i, shape=shape)

    temp_file = Path(tempfile.mktemp(suffix=".safetensors"))
    preprocessor.compute_all(save_path=temp_file)
    return temp_file


def benchmark_shape_access(cache_path, num_iterations=1000, batch_size=8):
    """Benchmark repeated get_shape() calls"""
    print(f"\nBenchmarking {num_iterations} iterations with batch_size={batch_size}")
    print("=" * 60)

    # Load dataset (this is when caching happens)
    load_start = time.time()
    dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")
    load_time = time.time() - load_start
    print(f"Dataset load time (with caching): {load_time:.4f}s")

    # Benchmark shape access
    num_samples = dataset.num_samples
    total_accesses = 0

    start = time.time()
    for iteration in range(num_iterations):
        # Simulate a training batch
        for _ in range(batch_size):
            idx = iteration % num_samples
            shape = dataset.get_shape(idx)
            total_accesses += 1

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Total shape accesses: {total_accesses}")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Average per access: {elapsed / total_accesses * 1000:.4f}ms")
    print(f"  Throughput: {total_accesses / elapsed:.1f} accesses/sec")

    return elapsed, total_accesses


def main():
    print("CDC Shape Cache Benchmark")
    print("=" * 60)

    # Create test cache
    cache_path = create_test_cache(num_samples=500, shape=(16, 64, 64))

    try:
        # Benchmark with typical training workload
        # Simulates 1000 training steps with batch_size=8
        benchmark_shape_access(cache_path, num_iterations=1000, batch_size=8)

        print("\n" + "=" * 60)
        print("Summary:")
        print("  With in-memory caching, shape access should be:")
        print("  - Sub-millisecond per access")
        print("  - No disk I/O after initial load")
        print("  - Constant time regardless of cache file size")

    finally:
        # Cleanup
        if cache_path.exists():
            cache_path.unlink()
            print(f"\nCleaned up test file: {cache_path}")


if __name__ == "__main__":
    main()
