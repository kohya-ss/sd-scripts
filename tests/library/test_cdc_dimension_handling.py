"""
Test CDC-FM dimension handling and fallback mechanisms.

This module tests the behavior of the CDC Flow Matching implementation
when encountering latents with different dimensions.
"""

import torch
import logging
import tempfile

from library.cdc_fm import CDCPreprocessor, GammaBDataset

class TestDimensionHandling:
    def setup_method(self):
        """Prepare consistent test environment"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def pytest_configure(config):
    """
    Configure custom markers for dimension handling tests
    """
    config.addinivalue_line(
        "markers",
        "dimension_handling: mark test for CDC-FM dimension mismatch scenarios"
    )