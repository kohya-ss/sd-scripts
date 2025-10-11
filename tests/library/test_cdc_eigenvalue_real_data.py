"""
Tests using realistic high-dimensional data to catch scaling bugs.

This test uses realistic VAE-like latents to ensure eigenvalue normalization
works correctly on real-world data.
"""

import numpy as np
import pytest
import torch
from safetensors import safe_open

from library.cdc_fm import CDCPreprocessor


class TestRealisticDataScaling:
    """Test eigenvalue scaling with realistic high-dimensional data"""

    def test_high_dimensional_latents_not_saturated(self, tmp_path):
        """
        Verify that high-dimensional realistic latents don't saturate eigenvalues.

        This test simulates real FLUX training data:
        - High dimension (16×64×64 = 65536)
        - Varied content (different variance in different regions)
        - Realistic magnitude (VAE output scale)
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create 20 samples with realistic varied structure
        for i in range(20):
            # High-dimensional latent like FLUX
            latent = torch.zeros(16, 64, 64, dtype=torch.float32)

            # Create varied structure across the latent
            # Different channels have different patterns (realistic for VAE)
            for c in range(16):
                # Some channels have gradients
                if c < 4:
                    for h in range(64):
                        for w in range(64):
                            latent[c, h, w] = (h + w) / 128.0
                # Some channels have patterns
                elif c < 8:
                    for h in range(64):
                        for w in range(64):
                            latent[c, h, w] = np.sin(h / 10.0) * np.cos(w / 10.0)
                # Some channels are more uniform
                else:
                    latent[c, :, :] = c * 0.1

            # Add per-sample variation (different "subjects")
            latent = latent * (1.0 + i * 0.2)

            # Add realistic VAE-like noise/variation
            latent = latent + torch.linspace(-0.5, 0.5, 16).view(16, 1, 1).expand(16, 64, 64) * (i % 3)

            metadata = {'image_key': f'test_image_{i}'}


            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_realistic_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify eigenvalues are NOT all saturated at 1.0
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(20):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                all_eigvals.extend(eigvals)

            all_eigvals = np.array(all_eigvals)
            non_zero_eigvals = all_eigvals[all_eigvals > 1e-6]

            # Critical: eigenvalues should NOT all be 1.0
            at_max = np.sum(np.abs(all_eigvals - 1.0) < 0.01)
            total = len(non_zero_eigvals)
            percent_at_max = (at_max / total * 100) if total > 0 else 0

            print(f"\n✓ Eigenvalue range: [{all_eigvals.min():.4f}, {all_eigvals.max():.4f}]")
            print(f"✓ Mean: {np.mean(non_zero_eigvals):.4f}")
            print(f"✓ Std: {np.std(non_zero_eigvals):.4f}")
            print(f"✓ At max (1.0): {at_max}/{total} ({percent_at_max:.1f}%)")

            # FAIL if too many eigenvalues are saturated at 1.0
            assert percent_at_max < 80, (
                f"{percent_at_max:.1f}% of eigenvalues are saturated at 1.0! "
                f"This indicates the normalization bug - raw eigenvalues are not being "
                f"scaled before clamping. Range: [{all_eigvals.min():.4f}, {all_eigvals.max():.4f}]"
            )

            # Should have good diversity
            assert np.std(non_zero_eigvals) > 0.1, (
                f"Eigenvalue std {np.std(non_zero_eigvals):.4f} is too low. "
                f"Should see diverse eigenvalues, not all the same value."
            )

            # Mean should be in reasonable range (not all 1.0)
            mean_eigval = np.mean(non_zero_eigvals)
            assert 0.05 < mean_eigval < 0.9, (
                f"Mean eigenvalue {mean_eigval:.4f} is outside expected range [0.05, 0.9]. "
                f"If mean ≈ 1.0, eigenvalues are saturated."
            )

    def test_eigenvalue_diversity_scales_with_data_variance(self, tmp_path):
        """
        Test that datasets with more variance produce more diverse eigenvalues.

        This ensures the normalization preserves relative information.
        """
        # Create two preprocessors with different data variance
        results = {}

        for variance_scale in [0.5, 2.0]:
            preprocessor = CDCPreprocessor(
                k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
            )

            for i in range(15):
                latent = torch.zeros(16, 32, 32, dtype=torch.float32)

                # Create varied patterns
                for c in range(16):
                    for h in range(32):
                        for w in range(32):
                            latent[c, h, w] = (
                                np.sin(h / 5.0 + i) * np.cos(w / 5.0 + c) * variance_scale
                            )

                metadata = {'image_key': f'test_image_{i}'}


                preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

            output_path = tmp_path / f"test_variance_{variance_scale}.safetensors"
            preprocessor.compute_all(save_path=output_path)

            with safe_open(str(output_path), framework="pt", device="cpu") as f:
                eigvals = []
                for i in range(15):
                    ev = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                    eigvals.extend(ev[ev > 1e-6])

                results[variance_scale] = {
                    'mean': np.mean(eigvals),
                    'std': np.std(eigvals),
                    'range': (np.min(eigvals), np.max(eigvals))
                }

        print(f"\n✓ Low variance data: mean={results[0.5]['mean']:.4f}, std={results[0.5]['std']:.4f}")
        print(f"✓ High variance data: mean={results[2.0]['mean']:.4f}, std={results[2.0]['std']:.4f}")

        # Both should have diversity (not saturated)
        for scale in [0.5, 2.0]:
            assert results[scale]['std'] > 0.1, (
                f"Variance scale {scale} has too low std: {results[scale]['std']:.4f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
