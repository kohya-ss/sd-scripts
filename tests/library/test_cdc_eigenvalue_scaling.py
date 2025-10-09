"""
Tests to verify CDC eigenvalue scaling is correct.

These tests ensure eigenvalues are properly scaled to prevent training loss explosion.
"""

import numpy as np
import pytest
import torch
from safetensors import safe_open

from library.cdc_fm import CDCPreprocessor


class TestEigenvalueScaling:
    """Test that eigenvalues are properly scaled to reasonable ranges"""

    def test_eigenvalues_in_correct_range(self, tmp_path):
        """Verify eigenvalues are scaled to ~0.01-1.0 range, not millions"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Add deterministic latents with structured patterns
        for i in range(10):
            # Create gradient pattern: values from 0 to 2.0 across spatial dims
            latent = torch.zeros(16, 8, 8, dtype=torch.float32)
            for h in range(8):
                for w in range(8):
                    latent[:, h, w] = (h * 8 + w) / 32.0  # Range [0, 2.0]
            # Add per-sample variation
            latent = latent + i * 0.1
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape)

        output_path = tmp_path / "test_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify eigenvalues are in correct range
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(10):
                eigvals = f.get_tensor(f"eigenvalues/{i}").numpy()
                all_eigvals.extend(eigvals)

            all_eigvals = np.array(all_eigvals)

            # Filter out zero eigenvalues (from padding when k < d_cdc)
            non_zero_eigvals = all_eigvals[all_eigvals > 1e-6]

            # Critical assertions for eigenvalue scale
            assert all_eigvals.max() < 10.0, f"Max eigenvalue {all_eigvals.max():.2e} is too large (should be <10)"
            assert len(non_zero_eigvals) > 0, "Should have some non-zero eigenvalues"
            assert np.mean(non_zero_eigvals) < 2.0, f"Mean eigenvalue {np.mean(non_zero_eigvals):.2e} is too large"

            # Check sqrt (used in noise) is reasonable
            sqrt_max = np.sqrt(all_eigvals.max())
            assert sqrt_max < 5.0, f"sqrt(max eigenvalue) = {sqrt_max:.2f} will cause noise explosion"

            print(f"\n✓ Eigenvalue range: [{all_eigvals.min():.4f}, {all_eigvals.max():.4f}]")
            print(f"✓ Non-zero eigenvalues: {len(non_zero_eigvals)}/{len(all_eigvals)}")
            print(f"✓ Mean (non-zero): {np.mean(non_zero_eigvals):.4f}")
            print(f"✓ sqrt(max): {sqrt_max:.4f}")

    def test_eigenvalues_not_all_zero(self, tmp_path):
        """Ensure eigenvalues are not all zero (indicating computation failure)"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        for i in range(10):
            # Create deterministic pattern
            latent = torch.zeros(16, 4, 4, dtype=torch.float32)
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c + h * 4 + w) / 32.0 + i * 0.2
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape)

        output_path = tmp_path / "test_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(10):
                eigvals = f.get_tensor(f"eigenvalues/{i}").numpy()
                all_eigvals.extend(eigvals)

            all_eigvals = np.array(all_eigvals)
            non_zero_eigvals = all_eigvals[all_eigvals > 1e-6]

            # With clamping, eigenvalues will be in range [1e-3, gamma*1.0]
            # Check that we have some non-zero eigenvalues
            assert len(non_zero_eigvals) > 0, "All eigenvalues are zero - computation failed"

            # Check they're in the expected clamped range
            assert np.all(non_zero_eigvals >= 1e-3), f"Some eigenvalues below clamp min: {np.min(non_zero_eigvals)}"
            assert np.all(non_zero_eigvals <= 1.0), f"Some eigenvalues above clamp max: {np.max(non_zero_eigvals)}"

            print(f"\n✓ Non-zero eigenvalues: {len(non_zero_eigvals)}/{len(all_eigvals)}")
            print(f"✓ Range: [{np.min(non_zero_eigvals):.4f}, {np.max(non_zero_eigvals):.4f}]")
            print(f"✓ Mean: {np.mean(non_zero_eigvals):.4f}")

    def test_fp16_storage_no_overflow(self, tmp_path):
        """Verify fp16 storage doesn't overflow (max fp16 = 65,504)"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        for i in range(10):
            # Create deterministic pattern with higher magnitude
            latent = torch.zeros(16, 8, 8, dtype=torch.float32)
            for h in range(8):
                for w in range(8):
                    latent[:, h, w] = (h * 8 + w) / 16.0  # Range [0, 4.0]
            latent = latent + i * 0.3
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape)

        output_path = tmp_path / "test_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            # Check dtype is fp16
            eigvecs = f.get_tensor("eigenvectors/0")
            eigvals = f.get_tensor("eigenvalues/0")

            assert eigvecs.dtype == torch.float16, f"Expected fp16, got {eigvecs.dtype}"
            assert eigvals.dtype == torch.float16, f"Expected fp16, got {eigvals.dtype}"

            # Check no values near fp16 max (would indicate overflow)
            FP16_MAX = 65504
            max_eigval = eigvals.max().item()

            assert max_eigval < 100, (
                f"Eigenvalue {max_eigval:.2e} is suspiciously large for fp16 storage. "
                f"May indicate overflow (fp16 max = {FP16_MAX})"
            )

            print(f"\n✓ Storage dtype: {eigvals.dtype}")
            print(f"✓ Max eigenvalue: {max_eigval:.4f} (safe for fp16)")

    def test_latent_magnitude_preserved(self, tmp_path):
        """Verify latent magnitude is preserved (no unwanted normalization)"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        # Store original latents with deterministic patterns
        original_latents = []
        for i in range(10):
            # Create structured pattern with known magnitude
            latent = torch.zeros(16, 4, 4, dtype=torch.float32)
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c * 0.1 + h * 0.2 + w * 0.3) + i * 0.5
            original_latents.append(latent.clone())
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape)

        # Compute original latent statistics
        orig_std = torch.stack(original_latents).std().item()

        output_path = tmp_path / "test_gamma_b.safetensors"
        preprocessor.compute_all(save_path=output_path)

        # The stored latents should preserve original magnitude
        stored_latents_std = np.std([s.latent for s in preprocessor.batcher.samples])

        # Should be similar to original (within 20% due to potential batching effects)
        assert 0.8 * orig_std < stored_latents_std < 1.2 * orig_std, (
            f"Stored latent std {stored_latents_std:.2f} differs too much from "
            f"original {orig_std:.2f}. Latent magnitude was not preserved."
        )

        print(f"\n✓ Original latent std: {orig_std:.2f}")
        print(f"✓ Stored latent std: {stored_latents_std:.2f}")


class TestTrainingLossScale:
    """Test that eigenvalues produce reasonable loss magnitudes"""

    def test_noise_magnitude_reasonable(self, tmp_path):
        """Verify CDC noise has reasonable magnitude for training"""
        from library.cdc_fm import GammaBDataset

        # Create CDC cache with deterministic data
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        for i in range(10):
            # Create deterministic pattern
            latent = torch.zeros(16, 4, 4, dtype=torch.float32)
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c + h + w) / 20.0 + i * 0.1
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape)

        output_path = tmp_path / "test_gamma_b.safetensors"
        cdc_path = preprocessor.compute_all(save_path=output_path)

        # Load and compute noise
        gamma_b = GammaBDataset(gamma_b_path=cdc_path, device="cpu")

        # Simulate training scenario with deterministic data
        batch_size = 3
        latents = torch.zeros(batch_size, 16, 4, 4)
        for b in range(batch_size):
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latents[b, c, h, w] = (b + c + h + w) / 24.0
        t = torch.tensor([0.5, 0.7, 0.9])  # Different timesteps
        indices = [0, 5, 9]

        eigvecs, eigvals = gamma_b.get_gamma_b_sqrt(indices)
        noise = gamma_b.compute_sigma_t_x(eigvecs, eigvals, latents, t)

        # Check noise magnitude
        noise_std = noise.std().item()
        latent_std = latents.std().item()

        # Noise should be similar magnitude to input latents (within 10x)
        ratio = noise_std / latent_std
        assert 0.1 < ratio < 10.0, (
            f"Noise std ({noise_std:.3f}) vs latent std ({latent_std:.3f}) "
            f"ratio {ratio:.2f} is too extreme. Will cause training instability."
        )

        # Simulated MSE loss should be reasonable
        simulated_loss = torch.mean((noise - latents) ** 2).item()
        assert simulated_loss < 100.0, (
            f"Simulated MSE loss {simulated_loss:.2f} is too high. "
            f"Should be O(0.1-1.0) for stable training."
        )

        print(f"\n✓ Noise/latent ratio: {ratio:.2f}")
        print(f"✓ Simulated MSE loss: {simulated_loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
