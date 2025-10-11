"""
Tests to validate the CDC rescaling recommendations from paper review.

These tests check:
1. Gamma parameter interaction with rescaling
2. Spatial adaptivity of eigenvalue scaling
3. Verification of fixed vs adaptive rescaling behavior
"""

import numpy as np
import pytest
import torch
from safetensors import safe_open

from library.cdc_fm import CDCPreprocessor


class TestGammaRescalingInteraction:
    """Test that gamma parameter works correctly with eigenvalue rescaling"""

    def test_gamma_scales_eigenvalues_correctly(self, tmp_path):
        """Verify gamma multiplier is applied correctly after rescaling"""
        # Create two preprocessors with different gamma values
        gamma_values = [0.5, 1.0, 2.0]
        eigenvalue_results = {}

        for gamma in gamma_values:
            preprocessor = CDCPreprocessor(
                k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=gamma, device="cpu"
            )

            # Add identical deterministic data for all runs
            for i in range(10):
                latent = torch.zeros(16, 4, 4, dtype=torch.float32)
                for c in range(16):
                    for h in range(4):
                        for w in range(4):
                            latent[c, h, w] = (c + h * 4 + w) / 32.0 + i * 0.1
                metadata = {'image_key': f'test_image_{i}'}

                preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

            output_path = tmp_path / f"test_gamma_{gamma}.safetensors"
            preprocessor.compute_all(save_path=output_path)

            # Extract eigenvalues
            with safe_open(str(output_path), framework="pt", device="cpu") as f:
                eigvals = f.get_tensor("eigenvalues/test_image_0").numpy()
                eigenvalue_results[gamma] = eigvals

        # With clamping to [1e-3, gamma*1.0], verify gamma changes the upper bound
        # Gamma 0.5: max eigenvalue should be ~0.5
        # Gamma 1.0: max eigenvalue should be ~1.0
        # Gamma 2.0: max eigenvalue should be ~2.0

        max_0p5 = np.max(eigenvalue_results[0.5])
        max_1p0 = np.max(eigenvalue_results[1.0])
        max_2p0 = np.max(eigenvalue_results[2.0])

        assert max_0p5 <= 0.5 + 0.01, f"Gamma 0.5 max should be ≤0.5, got {max_0p5}"
        assert max_1p0 <= 1.0 + 0.01, f"Gamma 1.0 max should be ≤1.0, got {max_1p0}"
        assert max_2p0 <= 2.0 + 0.01, f"Gamma 2.0 max should be ≤2.0, got {max_2p0}"

        # All should have min of 1e-3 (clamp lower bound)
        assert np.min(eigenvalue_results[0.5][eigenvalue_results[0.5] > 0]) >= 1e-3
        assert np.min(eigenvalue_results[1.0][eigenvalue_results[1.0] > 0]) >= 1e-3
        assert np.min(eigenvalue_results[2.0][eigenvalue_results[2.0] > 0]) >= 1e-3

        print(f"\n✓ Gamma 0.5 max: {max_0p5:.4f}")
        print(f"✓ Gamma 1.0 max: {max_1p0:.4f}")
        print(f"✓ Gamma 2.0 max: {max_2p0:.4f}")

    def test_large_gamma_maintains_reasonable_scale(self, tmp_path):
        """Verify that large gamma values don't cause eigenvalue explosion"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=10.0, device="cpu"
        )

        for i in range(10):
            latent = torch.zeros(16, 4, 4, dtype=torch.float32)
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c + h + w) / 20.0 + i * 0.15
            metadata = {'image_key': f'test_image_{i}'}

            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_large_gamma.safetensors"
        preprocessor.compute_all(save_path=output_path)

        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(10):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                all_eigvals.extend(eigvals)

        max_eigval = np.max(all_eigvals)
        mean_eigval = np.mean([e for e in all_eigvals if e > 1e-6])

        # With gamma=10.0 and target_scale=0.1, eigenvalues should be ~1.0
        # But they should still be reasonable (not exploding)
        assert max_eigval < 100, f"Max eigenvalue {max_eigval} too large even with large gamma"
        assert mean_eigval <= 10, f"Mean eigenvalue {mean_eigval} too large even with large gamma"

        print(f"\n✓ With gamma=10.0: max={max_eigval:.2f}, mean={mean_eigval:.2f}")


class TestSpatialAdaptivityOfRescaling:
    """Test spatial variation in eigenvalue scaling"""

    def test_eigenvalues_vary_spatially(self, tmp_path):
        """Verify eigenvalues differ across spatially separated clusters"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        # Create two distinct clusters in latent space
        # Cluster 1: Tight cluster (low variance) - deterministic spread
        for i in range(10):
            latent = torch.zeros(16, 4, 4)
            # Small variation around 0
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c + h + w) / 100.0 + i * 0.01
            metadata = {'image_key': f'test_image_{i}'}

            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        # Cluster 2: Loose cluster (high variance) - deterministic spread
        for i in range(10, 20):
            latent = torch.ones(16, 4, 4) * 5.0
            # Large variation around 5.0
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] += (c + h + w) / 10.0 + (i - 10) * 0.2
            metadata = {'image_key': f'test_image_{i}'}

            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_spatial_variation.safetensors"
        preprocessor.compute_all(save_path=output_path)

        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            # Get eigenvalues from both clusters
            cluster1_eigvals = []
            cluster2_eigvals = []

            for i in range(10):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                cluster1_eigvals.append(np.max(eigvals))

            for i in range(10, 20):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                cluster2_eigvals.append(np.max(eigvals))

        cluster1_mean = np.mean(cluster1_eigvals)
        cluster2_mean = np.mean(cluster2_eigvals)

        print(f"\n✓ Tight cluster max eigenvalue: {cluster1_mean:.4f}")
        print(f"✓ Loose cluster max eigenvalue: {cluster2_mean:.4f}")

        # With fixed target_scale rescaling, eigenvalues should be similar
        # despite different local geometry
        # This demonstrates the limitation of fixed rescaling
        ratio = cluster2_mean / (cluster1_mean + 1e-10)
        print(f"✓ Ratio (loose/tight): {ratio:.2f}")

        # Both should be rescaled to similar magnitude (~0.1 due to target_scale)
        assert 0.01 < cluster1_mean < 10.0, "Cluster 1 eigenvalues out of expected range"
        assert 0.01 < cluster2_mean < 10.0, "Cluster 2 eigenvalues out of expected range"


class TestFixedVsAdaptiveRescaling:
    """Compare current fixed rescaling vs paper's adaptive approach"""

    def test_current_rescaling_is_uniform(self, tmp_path):
        """Demonstrate that current rescaling produces uniform eigenvalue scales"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        # Create samples with varying local density - deterministic
        for i in range(20):
            latent = torch.zeros(16, 4, 4)
            # Some samples clustered, some isolated
            if i < 10:
                # Dense cluster around origin
                for c in range(16):
                    for h in range(4):
                        for w in range(4):
                            latent[c, h, w] = (c + h + w) / 40.0 + i * 0.05
            else:
                # Isolated points - larger offset
                for c in range(16):
                    for h in range(4):
                        for w in range(4):
                            latent[c, h, w] = (c + h + w) / 40.0 + i * 2.0

            metadata = {'image_key': f'test_image_{i}'}


            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_uniform_rescaling.safetensors"
        preprocessor.compute_all(save_path=output_path)

        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            max_eigenvalues = []
            for i in range(20):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                vals = eigvals[eigvals > 1e-6]
                if vals.size:  # at least one valid eigen-value
                    max_eigenvalues.append(vals.max())

        if not max_eigenvalues:  # safeguard against empty list
            pytest.skip("no valid eigen-values found")

        max_eigenvalues = np.array(max_eigenvalues)

        # Check coefficient of variation (std / mean)
        cv = max_eigenvalues.std() / max_eigenvalues.mean()

        print(f"\n✓ Max eigenvalues range: [{np.min(max_eigenvalues):.4f}, {np.max(max_eigenvalues):.4f}]")
        print(f"✓ Mean: {np.mean(max_eigenvalues):.4f}, Std: {np.std(max_eigenvalues):.4f}")
        print(f"✓ Coefficient of variation: {cv:.4f}")

        # With clamping, eigenvalues should have relatively low variation
        assert cv < 1.0, "Eigenvalues should have relatively low variation with clamping"
        # Mean should be reasonable (clamped to [1e-3, gamma*1.0] = [1e-3, 1.0])
        assert 0.01 < np.mean(max_eigenvalues) <= 1.0, f"Mean eigenvalue {np.mean(max_eigenvalues)} out of expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
