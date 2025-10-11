"""
Performance benchmarking for CDC Flow Matching implementation.

This module tests the computational overhead and noise injection properties
of the CDC-FM preprocessing pipeline.
"""

import time
import tempfile
import torch
import numpy as np
import pytest

from library.cdc_fm import CDCPreprocessor, GammaBDataset

class TestCDCPerformance:
    """
    Performance and Noise Injection Verification Tests for CDC Flow Matching

    These tests validate the computational performance and noise injection properties
    of the CDC-FM preprocessing pipeline across different latent sizes.

    Key Verification Points:
    1. Computational efficiency for various latent dimensions
    2. Noise injection statistical properties
    3. Eigenvector and eigenvalue characteristics
    """

    @pytest.fixture(params=[
        (3, 32, 32),   # Small latent: typical for compact representations
        (3, 64, 64),   # Medium latent: standard feature maps
        (3, 128, 128)  # Large latent: high-resolution feature spaces
    ])
    def latent_sizes(self, request):
        """
        Parametrized fixture generating test cases for different latent sizes.

        Rationale:
        - Tests robustness across various computational scales
        - Ensures consistent behavior from compact to large representations
        - Identifies potential dimensionality-related performance bottlenecks
        """
        return request.param

    def test_computational_overhead(self, latent_sizes):
        """
        Measure computational overhead of CDC preprocessing across latent sizes.

        Performance Verification Objectives:
        1. Verify preprocessing time scales predictably with input dimensions
        2. Ensure adaptive k-neighbors works efficiently
        3. Validate computational overhead remains within acceptable bounds

        Performance Metrics:
        - Total preprocessing time
        - Per-sample processing time
        - Computational complexity indicators

        Args:
            latent_sizes (tuple): Latent dimensions (C, H, W) to benchmark
        """
        # Tuned preprocessing configuration
        preprocessor = CDCPreprocessor(
            k_neighbors=256,       # Comprehensive neighborhood exploration
            d_cdc=8,               # Geometric embedding dimensionality
            debug=True,            # Enable detailed performance logging
            adaptive_k=True        # Dynamic neighborhood size adjustment
        )

        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)  # Consistent random generation

        # Generate representative latent batch
        batch_size = 32
        latents = torch.randn(batch_size, *latent_sizes)

        # Precision timing of preprocessing
        start_time = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
            # Add latents with traceable metadata
            for i, latent in enumerate(latents):
                preprocessor.add_latent(
                    latent,
                    global_idx=i,
                    metadata={'image_key': f'perf_test_image_{i}'}
                )

            # Compute CDC results
            cdc_path = preprocessor.compute_all(tmp_file.name)

        # Calculate precise preprocessing metrics
        end_time = time.perf_counter()
        preprocessing_time = end_time - start_time
        per_sample_time = preprocessing_time / batch_size

        # Performance reporting and assertions
        input_volume = np.prod(latent_sizes)
        time_complexity_indicator = preprocessing_time / input_volume

        print(f"\nPerformance Breakdown:")
        print(f"  Latent Size:       {latent_sizes}")
        print(f"  Total Samples:     {batch_size}")
        print(f"  Input Volume:      {input_volume}")
        print(f"  Total Time:        {preprocessing_time:.4f} seconds")
        print(f"  Per Sample Time:   {per_sample_time:.6f} seconds")
        print(f"  Time/Volume Ratio: {time_complexity_indicator:.8f} seconds/voxel")

        # Adaptive thresholds based on input dimensions
        max_total_time = 10.0  # Base threshold
        max_per_sample_time = 2.0  # Per-sample time threshold (more lenient)

        # Different time complexity thresholds for different latent sizes
        max_time_complexity = (
            1e-2 if np.prod(latent_sizes) <= 3072 else  # Smaller latents
            1e-4  # Standard latents
        )

        # Performance assertions with informative error messages
        assert preprocessing_time < max_total_time, (
            f"Total preprocessing time exceeded threshold!\n"
            f"  Latent Size:       {latent_sizes}\n"
            f"  Total Time:        {preprocessing_time:.4f} seconds\n"
            f"  Threshold:         {max_total_time} seconds"
        )

        assert per_sample_time < max_per_sample_time, (
            f"Per-sample processing time exceeded threshold!\n"
            f"  Latent Size:       {latent_sizes}\n"
            f"  Per Sample Time:   {per_sample_time:.6f} seconds\n"
            f"  Threshold:         {max_per_sample_time} seconds"
        )

        # More adaptable time complexity check
        assert time_complexity_indicator < max_time_complexity, (
            f"Time complexity scaling exceeded expectations!\n"
            f"  Latent Size:       {latent_sizes}\n"
            f"  Input Volume:      {input_volume}\n"
            f"  Time/Volume Ratio: {time_complexity_indicator:.8f} seconds/voxel\n"
            f"  Threshold:         {max_time_complexity} seconds/voxel"
        )

    def test_noise_distribution(self, latent_sizes):
        """
        Verify CDC noise injection quality and properties.

        Based on test plan objectives:
        1. CDC noise is actually being generated (not all Gaussian fallback)
        2. Eigenvalues are valid (non-negative, bounded)
        3. CDC components are finite and usable for noise generation

        Args:
            latent_sizes (tuple): Latent dimensions (C, H, W)
        """
        # Preprocessing configuration
        preprocessor = CDCPreprocessor(
            k_neighbors=16,  # Reduced to match batch size
            d_cdc=8,
            gamma=1.0,
            debug=True,
            adaptive_k=True
        )

        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)

        # Generate batch of latents
        batch_size = 32
        latents = torch.randn(batch_size, *latent_sizes)

        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
            # Add latents with metadata
            for i, latent in enumerate(latents):
                preprocessor.add_latent(
                    latent,
                    global_idx=i,
                    metadata={'image_key': f'noise_dist_image_{i}'}
                )

            # Compute CDC results
            cdc_path = preprocessor.compute_all(tmp_file.name)

            # Analyze noise properties
            dataset = GammaBDataset(cdc_path)

            # Track samples that used CDC vs Gaussian fallback
            cdc_samples = 0
            gaussian_samples = 0
            eigenvalue_stats = {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'sum': 0.0
            }

            # Verify each sample's CDC components
            for i in range(batch_size):
                image_key = f'noise_dist_image_{i}'

                # Get eigenvectors and eigenvalues
                eigvecs, eigvals = dataset.get_gamma_b_sqrt([image_key])

                # Skip zero eigenvectors (fallback case)
                if torch.all(eigvecs[0] == 0):
                    gaussian_samples += 1
                    continue

                # Get the top d_cdc eigenvectors and eigenvalues
                top_eigvecs = eigvecs[0]  # (d_cdc, d)
                top_eigvals = eigvals[0]  # (d_cdc,)

                # Basic validity checks
                assert torch.all(torch.isfinite(top_eigvecs)), f"Non-finite eigenvectors for sample {i}"
                assert torch.all(torch.isfinite(top_eigvals)), f"Non-finite eigenvalues for sample {i}"

                # Eigenvalue bounds (should be positive and <= 1.0 based on CDC-FM)
                assert torch.all(top_eigvals >= 0), f"Negative eigenvalues for sample {i}: {top_eigvals}"
                assert torch.all(top_eigvals <= 1.0), f"Eigenvalues exceed 1.0 for sample {i}: {top_eigvals}"

                # Update statistics
                eigenvalue_stats['min'] = min(eigenvalue_stats['min'], top_eigvals.min().item())
                eigenvalue_stats['max'] = max(eigenvalue_stats['max'], top_eigvals.max().item())
                eigenvalue_stats['sum'] += top_eigvals.sum().item()

                cdc_samples += 1

            # Compute mean eigenvalue across all CDC samples
            if cdc_samples > 0:
                eigenvalue_stats['mean'] = eigenvalue_stats['sum'] / (cdc_samples * 8)  # 8 = d_cdc

            # Print final statistics
            print(f"\nNoise Distribution Results for latent size {latent_sizes}:")
            print(f"  CDC samples:        {cdc_samples}/{batch_size}")
            print(f"  Gaussian fallback:  {gaussian_samples}/{batch_size}")
            print(f"  Eigenvalue min:     {eigenvalue_stats['min']:.4f}")
            print(f"  Eigenvalue max:     {eigenvalue_stats['max']:.4f}")
            print(f"  Eigenvalue mean:    {eigenvalue_stats['mean']:.4f}")

            # Assertions based on plan objectives

            # 1. CDC noise should be generated for most samples
            assert cdc_samples > 0, "No samples used CDC noise injection"
            assert gaussian_samples < batch_size // 2, (
                f"Too many samples fell back to Gaussian noise: {gaussian_samples}/{batch_size}"
            )

            # 2. Eigenvalues should be valid (non-negative and bounded)
            assert eigenvalue_stats['min'] >= 0, "Eigenvalues should be non-negative"
            assert eigenvalue_stats['max'] <= 1.0, "Maximum eigenvalue exceeds 1.0"

            # 3. Mean eigenvalue should be reasonable (not degenerate)
            assert eigenvalue_stats['mean'] > 0.05, (
                f"Mean eigenvalue too low ({eigenvalue_stats['mean']:.4f}), "
                "suggests degenerate CDC components"
            )

def pytest_configure(config):
    """
    Configure performance benchmarking markers
    """
    config.addinivalue_line(
        "markers",
        "performance: mark test to verify CDC-FM computational performance"
    )
    config.addinivalue_line(
        "markers",
        "noise_distribution: mark test to verify noise injection properties"
    )