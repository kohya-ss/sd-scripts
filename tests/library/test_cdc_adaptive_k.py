"""
Test adaptive k_neighbors functionality in CDC-FM.

Verifies that adaptive k properly adjusts based on bucket sizes.
"""

import pytest
import torch

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestAdaptiveK:
    """Test adaptive k_neighbors behavior"""

    @pytest.fixture
    def temp_cache_path(self, tmp_path):
        """Create temporary cache path"""
        return tmp_path / "adaptive_k_test.safetensors"

    def test_fixed_k_skips_small_buckets(self, temp_cache_path):
        """
        Test that fixed k mode skips buckets with < k_neighbors samples.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=32,
            k_bandwidth=8,
            d_cdc=4,
            gamma=1.0,
            device='cpu',
            debug=False,
            adaptive_k=False  # Fixed mode
        )

        # Add 10 samples (< k=32, should be skipped)
        shape = (4, 16, 16)
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                shape=shape,
                metadata={'image_key': f'test_{i}'}
            )

        preprocessor.compute_all(temp_cache_path)

        # Load and verify zeros (Gaussian fallback)
        dataset = GammaBDataset(gamma_b_path=temp_cache_path, device='cpu')
        eigvecs, eigvals = dataset.get_gamma_b_sqrt(['test_0'], device='cpu')

        # Should be all zeros (fallback)
        assert torch.allclose(eigvecs, torch.zeros_like(eigvecs), atol=1e-6)
        assert torch.allclose(eigvals, torch.zeros_like(eigvals), atol=1e-6)

    def test_adaptive_k_uses_available_neighbors(self, temp_cache_path):
        """
        Test that adaptive k mode uses k=bucket_size-1 for small buckets.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=32,
            k_bandwidth=8,
            d_cdc=4,
            gamma=1.0,
            device='cpu',
            debug=False,
            adaptive_k=True,
            min_bucket_size=8
        )

        # Add 20 samples (< k=32, should use k=19)
        shape = (4, 16, 16)
        for i in range(20):
            latent = torch.randn(*shape, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                shape=shape,
                metadata={'image_key': f'test_{i}'}
            )

        preprocessor.compute_all(temp_cache_path)

        # Load and verify non-zero (CDC computed)
        dataset = GammaBDataset(gamma_b_path=temp_cache_path, device='cpu')
        eigvecs, eigvals = dataset.get_gamma_b_sqrt(['test_0'], device='cpu')

        # Should NOT be all zeros (CDC was computed)
        assert not torch.allclose(eigvecs, torch.zeros_like(eigvecs), atol=1e-6)
        assert not torch.allclose(eigvals, torch.zeros_like(eigvals), atol=1e-6)

    def test_adaptive_k_respects_min_bucket_size(self, temp_cache_path):
        """
        Test that adaptive k mode skips buckets below min_bucket_size.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=32,
            k_bandwidth=8,
            d_cdc=4,
            gamma=1.0,
            device='cpu',
            debug=False,
            adaptive_k=True,
            min_bucket_size=16
        )

        # Add 10 samples (< min_bucket_size=16, should be skipped)
        shape = (4, 16, 16)
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                shape=shape,
                metadata={'image_key': f'test_{i}'}
            )

        preprocessor.compute_all(temp_cache_path)

        # Load and verify zeros (skipped due to min_bucket_size)
        dataset = GammaBDataset(gamma_b_path=temp_cache_path, device='cpu')
        eigvecs, eigvals = dataset.get_gamma_b_sqrt(['test_0'], device='cpu')

        # Should be all zeros (skipped)
        assert torch.allclose(eigvecs, torch.zeros_like(eigvecs), atol=1e-6)
        assert torch.allclose(eigvals, torch.zeros_like(eigvals), atol=1e-6)

    def test_adaptive_k_mixed_bucket_sizes(self, temp_cache_path):
        """
        Test adaptive k with multiple buckets of different sizes.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=32,
            k_bandwidth=8,
            d_cdc=4,
            gamma=1.0,
            device='cpu',
            debug=False,
            adaptive_k=True,
            min_bucket_size=8
        )

        # Bucket 1: 10 samples (adaptive k=9)
        for i in range(10):
            latent = torch.randn(4, 16, 16, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                shape=(4, 16, 16),
                metadata={'image_key': f'small_{i}'}
            )

        # Bucket 2: 40 samples (full k=32)
        for i in range(40):
            latent = torch.randn(4, 32, 32, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=100+i,
                shape=(4, 32, 32),
                metadata={'image_key': f'large_{i}'}
            )

        # Bucket 3: 5 samples (< min=8, skipped)
        for i in range(5):
            latent = torch.randn(4, 8, 8, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=200+i,
                shape=(4, 8, 8),
                metadata={'image_key': f'tiny_{i}'}
            )

        preprocessor.compute_all(temp_cache_path)
        dataset = GammaBDataset(gamma_b_path=temp_cache_path, device='cpu')

        # Bucket 1: Should have CDC (non-zero)
        eigvecs_small, eigvals_small = dataset.get_gamma_b_sqrt(['small_0'], device='cpu')
        assert not torch.allclose(eigvecs_small, torch.zeros_like(eigvecs_small), atol=1e-6)

        # Bucket 2: Should have CDC (non-zero)
        eigvecs_large, eigvals_large = dataset.get_gamma_b_sqrt(['large_0'], device='cpu')
        assert not torch.allclose(eigvecs_large, torch.zeros_like(eigvecs_large), atol=1e-6)

        # Bucket 3: Should be skipped (zeros)
        eigvecs_tiny, eigvals_tiny = dataset.get_gamma_b_sqrt(['tiny_0'], device='cpu')
        assert torch.allclose(eigvecs_tiny, torch.zeros_like(eigvecs_tiny), atol=1e-6)
        assert torch.allclose(eigvals_tiny, torch.zeros_like(eigvals_tiny), atol=1e-6)

    def test_adaptive_k_uses_full_k_when_available(self, temp_cache_path):
        """
        Test that adaptive k uses full k_neighbors when bucket is large enough.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=16,
            k_bandwidth=4,
            d_cdc=4,
            gamma=1.0,
            device='cpu',
            debug=False,
            adaptive_k=True,
            min_bucket_size=8
        )

        # Add 50 samples (> k=16, should use full k=16)
        shape = (4, 16, 16)
        for i in range(50):
            latent = torch.randn(*shape, dtype=torch.float32).numpy()
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                shape=shape,
                metadata={'image_key': f'test_{i}'}
            )

        preprocessor.compute_all(temp_cache_path)

        # Load and verify CDC was computed
        dataset = GammaBDataset(gamma_b_path=temp_cache_path, device='cpu')
        eigvecs, eigvals = dataset.get_gamma_b_sqrt(['test_0'], device='cpu')

        # Should have non-zero eigenvalues
        assert not torch.allclose(eigvals, torch.zeros_like(eigvals), atol=1e-6)
        # Eigenvalues should be positive
        assert (eigvals >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
