"""
Performance and Interpolation Tests for CDC Flow Matching

This module provides testing of:
1. Computational overhead
2. Noise injection properties
3. Interpolation vs. pad/truncate methods
4. Spatial structure preservation
"""

import pytest
import torch
import time
import tempfile
import numpy as np
import torch.nn.functional as F

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestCDCPerformanceAndInterpolation:
    """
    Comprehensive performance testing for CDC Flow Matching
    Covers computational efficiency, noise properties, and interpolation quality
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
        """
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

    def test_interpolation_reconstruction(self):
        """
        Compare interpolation vs pad/truncate reconstruction methods for CDC.
        """
        # Create test latents with different sizes - deterministic
        latent_small = torch.zeros(16, 4, 4)
        for c in range(16):
            for h in range(4):
                for w in range(4):
                    latent_small[c, h, w] = (c * 0.1 + h * 0.2 + w * 0.3) / 3.0

        latent_large = torch.zeros(16, 8, 8)
        for c in range(16):
            for h in range(8):
                for w in range(8):
                    latent_large[c, h, w] = (c * 0.1 + h * 0.15 + w * 0.15) / 3.0

        target_h, target_w = 6, 6  # Median size

        # Method 1: Interpolation
        def interpolate_method(latent, target_h, target_w):
            latent_input = latent.unsqueeze(0)  # (1, C, H, W)
            latent_resized = F.interpolate(
                latent_input, size=(target_h, target_w), mode='bilinear', align_corners=False
            )
            # Resize back
            C, H, W = latent.shape
            latent_reconstructed = F.interpolate(
                latent_resized, size=(H, W), mode='bilinear', align_corners=False
            )
            error = torch.mean(torch.abs(latent_reconstructed - latent_input)).item()
            relative_error = error / (torch.mean(torch.abs(latent_input)).item() + 1e-8)
            return relative_error

        # Method 2: Pad/Truncate
        def pad_truncate_method(latent, target_h, target_w):
            C, H, W = latent.shape
            latent_flat = latent.reshape(-1)
            target_dim = C * target_h * target_w
            current_dim = C * H * W

            if current_dim == target_dim:
                latent_resized_flat = latent_flat
            elif current_dim > target_dim:
                # Truncate
                latent_resized_flat = latent_flat[:target_dim]
            else:
                # Pad
                latent_resized_flat = torch.zeros(target_dim)
                latent_resized_flat[:current_dim] = latent_flat

            # Resize back
            if current_dim == target_dim:
                latent_reconstructed_flat = latent_resized_flat
            elif current_dim > target_dim:
                # Pad back
                latent_reconstructed_flat = torch.zeros(current_dim)
                latent_reconstructed_flat[:target_dim] = latent_resized_flat
            else:
                # Truncate back
                latent_reconstructed_flat = latent_resized_flat[:current_dim]

            latent_reconstructed = latent_reconstructed_flat.reshape(C, H, W)
            error = torch.mean(torch.abs(latent_reconstructed - latent)).item()
            relative_error = error / (torch.mean(torch.abs(latent)).item() + 1e-8)
            return relative_error

        # Compare for small latent (needs padding)
        interp_error_small = interpolate_method(latent_small, target_h, target_w)
        pad_error_small = pad_truncate_method(latent_small, target_h, target_w)

        # Compare for large latent (needs truncation)
        interp_error_large = interpolate_method(latent_large, target_h, target_w)
        truncate_error_large = pad_truncate_method(latent_large, target_h, target_w)

        print("\n" + "=" * 60)
        print("Reconstruction Error Comparison")
        print("=" * 60)
        print("\nSmall latent (16x4x4 -> 16x6x6 -> 16x4x4):")
        print(f"  Interpolation error: {interp_error_small:.6f}")
        print(f"  Pad/truncate error:  {pad_error_small:.6f}")
        if pad_error_small > 0:
            print(f"  Improvement:         {(pad_error_small - interp_error_small) / pad_error_small * 100:.2f}%")
        else:
            print("  Note: Pad/truncate has 0 reconstruction error (perfect recovery)")
            print("        BUT the intermediate representation is corrupted with zeros!")

        print("\nLarge latent (16x8x8 -> 16x6x6 -> 16x8x8):")
        print(f"  Interpolation error: {interp_error_large:.6f}")
        print(f"  Pad/truncate error:  {truncate_error_large:.6f}")
        if truncate_error_large > 0:
            print(f"  Improvement:         {(truncate_error_large - interp_error_large) / truncate_error_large * 100:.2f}%")

        print("\nKey insight: For CDC, intermediate representation quality matters,")
        print("not reconstruction error. Interpolation preserves spatial structure.")

        # Verify interpolation errors are reasonable
        assert interp_error_small < 1.0, "Interpolation should have reasonable error"
        assert interp_error_large < 1.0, "Interpolation should have reasonable error"

    def test_spatial_structure_preservation(self):
        """
        Test that interpolation preserves spatial structure better than pad/truncate.
        """
        # Create a latent with clear spatial pattern (gradient)
        C, H, W = 16, 4, 4
        latent = torch.zeros(C, H, W)
        for i in range(H):
            for j in range(W):
                latent[:, i, j] = i * W + j  # Gradient pattern

        target_h, target_w = 6, 6

        # Interpolation
        latent_input = latent.unsqueeze(0)
        latent_interp = F.interpolate(
            latent_input, size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Pad/truncate
        latent_flat = latent.reshape(-1)
        target_dim = C * target_h * target_w
        latent_padded = torch.zeros(target_dim)
        latent_padded[:len(latent_flat)] = latent_flat
        latent_pad = latent_padded.reshape(C, target_h, target_w)

        # Check gradient preservation
        # For interpolation, adjacent pixels should have smooth gradients
        grad_x_interp = torch.abs(latent_interp[:, :, 1:] - latent_interp[:, :, :-1]).mean()
        grad_y_interp = torch.abs(latent_interp[:, 1:, :] - latent_interp[:, :-1, :]).mean()

        # For padding, there will be abrupt changes (gradient to zero)
        grad_x_pad = torch.abs(latent_pad[:, :, 1:] - latent_pad[:, :, :-1]).mean()
        grad_y_pad = torch.abs(latent_pad[:, 1:, :] - latent_pad[:, :-1, :]).mean()

        print("\n" + "=" * 60)
        print("Spatial Structure Preservation")
        print("=" * 60)
        print("\nGradient smoothness (lower is smoother):")
        print(f"  Interpolation - X gradient: {grad_x_interp:.4f}, Y gradient: {grad_y_interp:.4f}")
        print(f"  Pad/truncate  - X gradient: {grad_x_pad:.4f}, Y gradient: {grad_y_pad:.4f}")

        # Padding introduces larger gradients due to abrupt zeros
        assert grad_x_pad > grad_x_interp, "Padding should introduce larger gradients"
        assert grad_y_pad > grad_y_interp, "Padding should introduce larger gradients"


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
    config.addinivalue_line(
        "markers",
        "interpolation: mark test to verify interpolation quality"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])