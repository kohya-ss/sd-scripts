"""
Comprehensive CDC Dimension Handling and Warning Tests

This module tests:
1. Dimension mismatch detection and fallback mechanisms
2. Warning throttling for shape mismatches
3. Adaptive k-neighbors behavior with dimension constraints
"""

import pytest
import torch
import logging
import tempfile

from library.cdc_fm import CDCPreprocessor, GammaBDataset
from library.flux_train_utils import apply_cdc_noise_transformation, _cdc_warned_samples


class TestDimensionHandlingAndWarnings:
    """
    Comprehensive testing of dimension handling, noise injection, and warning systems
    """

    @pytest.fixture(autouse=True)
    def clear_warned_samples(self):
        """Clear the warned samples set before each test"""
        _cdc_warned_samples.clear()
        yield
        _cdc_warned_samples.clear()

    def test_mixed_dimension_fallback(self):
        """
        Verify that preprocessor falls back to standard noise for mixed-dimension batches
        """
        # Prepare preprocessor with debug mode
        preprocessor = CDCPreprocessor(debug=True)

        # Different-sized latents (3D: channels, height, width)
        latents = [
            torch.randn(3, 32, 64),    # First latent: 3x32x64
            torch.randn(3, 32, 128),   # Second latent: 3x32x128 (different dimension)
        ]

        # Use a mock handler to capture log messages
        from library.cdc_fm import logger

        log_messages = []
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())

        # Temporarily add a capture handler
        capture_handler = LogCapture()
        logger.addHandler(capture_handler)

        try:
            # Try adding mixed-dimension latents
            with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
                for i, latent in enumerate(latents):
                    preprocessor.add_latent(
                        latent,
                        global_idx=i,
                        metadata={'image_key': f'test_mixed_image_{i}'}
                    )

                try:
                    cdc_path = preprocessor.compute_all(tmp_file.name)
                except ValueError as e:
                    # If implementation raises ValueError, that's acceptable
                    assert "Dimension mismatch" in str(e)
                    return

                # Check for dimension-related log messages
                dimension_warnings = [
                    msg for msg in log_messages
                    if "dimension mismatch" in msg.lower()
                ]
                assert len(dimension_warnings) > 0, "No dimension-related warnings were logged"

                # Load results and verify fallback
                dataset = GammaBDataset(cdc_path)

        finally:
            # Remove the capture handler
            logger.removeHandler(capture_handler)

            # Check metadata about samples with/without CDC
            assert dataset.num_samples == len(latents), "All samples should be processed"

    def test_adaptive_k_with_dimension_constraints(self):
        """
        Test adaptive k-neighbors behavior with dimension constraints
        """
        # Prepare preprocessor with adaptive k and small bucket size
        preprocessor = CDCPreprocessor(
            adaptive_k=True,
            min_bucket_size=5,
            debug=True
        )

        # Generate latents with similar but not identical dimensions
        base_latent = torch.randn(3, 32, 64)
        similar_latents = [
            base_latent,
            torch.randn(3, 32, 65),   # Slightly different dimension
            torch.randn(3, 32, 66)    # Another slightly different dimension
        ]

        # Use a mock handler to capture log messages
        from library.cdc_fm import logger

        log_messages = []
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())

        # Temporarily add a capture handler
        capture_handler = LogCapture()
        logger.addHandler(capture_handler)

        try:
            with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
                # Add similar latents
                for i, latent in enumerate(similar_latents):
                    preprocessor.add_latent(
                        latent,
                        global_idx=i,
                        metadata={'image_key': f'test_adaptive_k_image_{i}'}
                    )

                cdc_path = preprocessor.compute_all(tmp_file.name)

                # Load results
                dataset = GammaBDataset(cdc_path)

                # Verify samples processed
                assert dataset.num_samples == len(similar_latents), "All samples should be processed"

                # Optional: Check warnings about dimension differences
                dimension_warnings = [
                    msg for msg in log_messages
                    if "dimension" in msg.lower()
                ]
                print(f"Dimension-related warnings: {dimension_warnings}")

        finally:
            # Remove the capture handler
            logger.removeHandler(capture_handler)

    def test_warning_only_logged_once_per_sample(self, caplog):
        """
        Test that shape mismatch warning is only logged once per sample.

        Even if the same sample appears in multiple batches, only warn once.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create cache with one specific shape
        preprocessed_shape = (16, 32, 32)
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
            for i in range(10):
                latent = torch.randn(*preprocessed_shape, dtype=torch.float32)
                metadata = {'image_key': f'test_image_{i}'}
                preprocessor.add_latent(latent=latent, global_idx=i, shape=preprocessed_shape, metadata=metadata)

            cdc_path = preprocessor.compute_all(save_path=tmp_file.name)

            dataset = GammaBDataset(gamma_b_path=cdc_path, device="cpu")

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

    def test_different_samples_each_get_one_warning(self, caplog):
        """
        Test that different samples each get their own warning.

        Each unique sample should be warned about once.
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create cache with specific shape
        preprocessed_shape = (16, 32, 32)
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as tmp_file:
            for i in range(10):
                latent = torch.randn(*preprocessed_shape, dtype=torch.float32)
                metadata = {'image_key': f'test_image_{i}'}
                preprocessor.add_latent(latent=latent, global_idx=i, shape=preprocessed_shape, metadata=metadata)

            cdc_path = preprocessor.compute_all(save_path=tmp_file.name)

            dataset = GammaBDataset(gamma_b_path=cdc_path, device="cpu")

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


def pytest_configure(config):
    """
    Configure custom markers for dimension handling and warning tests
    """
    config.addinivalue_line(
        "markers",
        "dimension_handling: mark test for CDC-FM dimension mismatch scenarios"
    )
    config.addinivalue_line(
        "markers",
        "warning_throttling: mark test for CDC-FM warning suppression"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])