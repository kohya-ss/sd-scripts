"""
Test warning throttling for CDC shape mismatches.

Ensures that duplicate warnings for the same sample are not logged repeatedly.
"""

import pytest
import torch
import logging
from pathlib import Path

from library.cdc_fm import CDCPreprocessor, GammaBDataset
from library.flux_train_utils import apply_cdc_noise_transformation, _cdc_warned_samples


class TestWarningThrottling:
    """Test that shape mismatch warnings are throttled"""

    @pytest.fixture(autouse=True)
    def clear_warned_samples(self):
        """Clear the warned samples set before each test"""
        _cdc_warned_samples.clear()
        yield
        _cdc_warned_samples.clear()

    @pytest.fixture
    def cdc_cache(self, tmp_path):
        """Create a test CDC cache with one shape"""
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create cache with one specific shape
        preprocessed_shape = (16, 32, 32)
        for i in range(10):
            latent = torch.randn(*preprocessed_shape, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=preprocessed_shape, metadata=metadata)

        cache_path = tmp_path / "test_throttle.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        return cache_path

    def test_warning_only_logged_once_per_sample(self, cdc_cache, caplog):
        """
        Test that shape mismatch warning is only logged once per sample.

        Even if the same sample appears in multiple batches, only warn once.
        """
        dataset = GammaBDataset(gamma_b_path=cdc_cache, device="cpu")

        # Use different shape at runtime to trigger mismatch
        runtime_shape = (16, 64, 64)
        timesteps = torch.tensor([100.0], dtype=torch.float32)
        image_keys = ['test_image_0']  # Same sample

        # First call - should warn
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise1 = torch.randn(1, *runtime_shape, dtype=torch.float32)
            _ = apply_cdc_noise_transformation(
                noise=noise1,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            # Should have exactly one warning
            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 1, "First call should produce exactly one warning"
            assert "CDC shape mismatch" in warnings[0].message

        # Second call with same sample - should NOT warn
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise2 = torch.randn(1, *runtime_shape, dtype=torch.float32)
            _ = apply_cdc_noise_transformation(
                noise=noise2,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            # Should have NO warnings
            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 0, "Second call with same sample should not warn"

        # Third call with same sample - still should NOT warn
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise3 = torch.randn(1, *runtime_shape, dtype=torch.float32)
            _ = apply_cdc_noise_transformation(
                noise=noise3,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 0, "Third call should still not warn"

    def test_different_samples_each_get_one_warning(self, cdc_cache, caplog):
        """
        Test that different samples each get their own warning.

        Each unique sample should be warned about once.
        """
        dataset = GammaBDataset(gamma_b_path=cdc_cache, device="cpu")

        runtime_shape = (16, 64, 64)
        timesteps = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)

        # First batch: samples 0, 1, 2
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise = torch.randn(3, *runtime_shape, dtype=torch.float32)
            image_keys = ['test_image_0', 'test_image_1', 'test_image_2']

            _ = apply_cdc_noise_transformation(
                noise=noise,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            # Should have 3 warnings (one per sample)
            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 3, "Should warn for each of the 3 samples"

        # Second batch: same samples 0, 1, 2
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise = torch.randn(3, *runtime_shape, dtype=torch.float32)
            image_keys = ['test_image_0', 'test_image_1', 'test_image_2']

            _ = apply_cdc_noise_transformation(
                noise=noise,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            # Should have NO warnings (already warned)
            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 0, "Should not warn again for same samples"

        # Third batch: new samples 3, 4
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            noise = torch.randn(2, *runtime_shape, dtype=torch.float32)
            image_keys = ['test_image_3', 'test_image_4']
            timesteps = torch.tensor([100.0, 200.0], dtype=torch.float32)

            _ = apply_cdc_noise_transformation(
                noise=noise,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
                device="cpu"
            )

            # Should have 2 warnings (new samples)
            warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
            assert len(warnings) == 2, "Should warn for each of the 2 new samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
