"""
Comprehensive CDC Eigenvalue Validation Tests

These tests ensure that eigenvalue computation and scaling work correctly
across various scenarios, including:
- Scaling to reasonable ranges
- Handling high-dimensional data
- Preserving latent information
- Preventing computational artifacts
"""

import numpy as np
import pytest
import torch
from safetensors import safe_open

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestEigenvalueScaling:
    """Verify eigenvalue scaling and computational properties"""

    def test_eigenvalues_in_correct_range(self, tmp_path):
        """
        Verify eigenvalues are scaled to ~0.01-1.0 range, not millions.

        Ensures:
        - No numerical explosions
        - Reasonable eigenvalue magnitudes
        - Consistent scaling across samples
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create deterministic latents with structured patterns
        for i in range(10):
            latent = torch.zeros(16, 8, 8, dtype=torch.float32)
            for h in range(8):
                for w in range(8):
                    latent[:, h, w] = (h * 8 + w) / 32.0  # Range [0, 2.0]
            latent = latent + i * 0.1
            metadata = {'image_key': f'test_image_{i}'}

            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify eigenvalues are in correct range
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(10):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                all_eigvals.extend(eigvals)

            all_eigvals = np.array(all_eigvals)
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

    def test_high_dimensional_latents_scaling(self, tmp_path):
        """
        Verify scaling for high-dimensional realistic latents.

        Key scenarios:
        - High-dimensional data (16×64×64)
        - Varied channel structures
        - Realistic VAE-like data
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create 20 samples with realistic varied structure
        for i in range(20):
            # High-dimensional latent like FLUX
            latent = torch.zeros(16, 64, 64, dtype=torch.float32)

            # Create varied structure across the latent
            for c in range(16):
                # Different patterns across channels
                if c < 4:
                    for h in range(64):
                        for w in range(64):
                            latent[c, h, w] = (h + w) / 128.0
                elif c < 8:
                    for h in range(64):
                        for w in range(64):
                            latent[c, h, w] = np.sin(h / 10.0) * np.cos(w / 10.0)
                else:
                    latent[c, :, :] = c * 0.1

            # Add per-sample variation
            latent = latent * (1.0 + i * 0.2)
            latent = latent + torch.linspace(-0.5, 0.5, 16).view(16, 1, 1).expand(16, 64, 64) * (i % 3)

            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "test_realistic_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify eigenvalues are not all saturated
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            all_eigvals = []
            for i in range(20):
                eigvals = f.get_tensor(f"eigenvalues/test_image_{i}").numpy()
                all_eigvals.extend(eigvals)

            all_eigvals = np.array(all_eigvals)
            non_zero_eigvals = all_eigvals[all_eigvals > 1e-6]

            at_max = np.sum(np.abs(all_eigvals - 1.0) < 0.01)
            total = len(non_zero_eigvals)
            percent_at_max = (at_max / total * 100) if total > 0 else 0

            print(f"\n✓ Eigenvalue range: [{all_eigvals.min():.4f}, {all_eigvals.max():.4f}]")
            print(f"✓ Mean: {np.mean(non_zero_eigvals):.4f}")
            print(f"✓ Std: {np.std(non_zero_eigvals):.4f}")
            print(f"✓ At max (1.0): {at_max}/{total} ({percent_at_max:.1f}%)")

            # Fail if too many eigenvalues are saturated
            assert percent_at_max < 80, (
                f"{percent_at_max:.1f}% of eigenvalues are saturated at 1.0! "
                f"Raw eigenvalues not scaled before clamping. "
                f"Range: [{all_eigvals.min():.4f}, {all_eigvals.max():.4f}]"
            )

            # Should have good diversity
            assert np.std(non_zero_eigvals) > 0.1, (
                f"Eigenvalue std {np.std(non_zero_eigvals):.4f} is too low. "
                f"Should see diverse eigenvalues, not all the same."
            )

            # Mean should be in reasonable range
            mean_eigval = np.mean(non_zero_eigvals)
            assert 0.05 < mean_eigval < 0.9, (
                f"Mean eigenvalue {mean_eigval:.4f} is outside expected range [0.05, 0.9]. "
                f"If mean ≈ 1.0, eigenvalues are saturated."
            )

    def test_noise_magnitude_reasonable(self, tmp_path):
        """
        Verify CDC noise has reasonable magnitude for training.

        Ensures noise:
        - Has similar scale to input latents
        - Won't destabilize training
        - Preserves input variance
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        for i in range(10):
            latent = torch.zeros(16, 4, 4, dtype=torch.float32)
            for c in range(16):
                for h in range(4):
                    for w in range(4):
                        latent[c, h, w] = (c + h + w) / 20.0 + i * 0.1
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

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
        image_keys = ['test_image_0', 'test_image_5', 'test_image_9']

        eigvecs, eigvals = gamma_b.get_gamma_b_sqrt(image_keys)
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