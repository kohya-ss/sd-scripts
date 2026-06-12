import pytest
import torch
from unittest.mock import Mock

from library.strategy_sd import SdTextEncodingStrategy


class TestSdTextEncodingStrategy:
    @pytest.fixture
    def strategy(self):
        """Create strategy instance with default settings."""
        return SdTextEncodingStrategy(clip_skip=None)

    @pytest.fixture
    def strategy_with_clip_skip(self):
        """Create strategy instance with CLIP skip enabled."""
        return SdTextEncodingStrategy(clip_skip=2)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.model_max_length = 77
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = 2
        tokenizer.eos_token_id = 2
        return tokenizer

    @pytest.fixture
    def mock_text_encoder(self):
        """Create a mock text encoder."""
        encoder = Mock()
        encoder.device = torch.device("cpu")

        def encode_side_effect(tokens, output_hidden_states=False, return_dict=False):
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            hidden_size = 768

            # Create deterministic hidden states
            hidden_state = torch.ones(batch_size, seq_len, hidden_size) * 0.5

            if return_dict:
                result = {
                    "hidden_states": [
                        hidden_state * 0.8,
                        hidden_state * 0.9,
                        hidden_state * 1.0,
                    ]
                }
                return result
            else:
                return [hidden_state]

        encoder.side_effect = encode_side_effect
        encoder.text_model = Mock()
        encoder.text_model.final_layer_norm = lambda x: x

        return encoder

    @pytest.fixture
    def mock_tokenize_strategy(self, mock_tokenizer):
        """Create a mock tokenize strategy."""
        strategy = Mock()
        strategy.tokenizer = mock_tokenizer
        return strategy

    # Test _encode_with_clip_skip
    def test_encode_without_clip_skip(self, strategy, mock_text_encoder):
        """Test encoding without CLIP skip."""
        tokens = torch.arange(154).reshape(2, 77)
        result = strategy._encode_with_clip_skip(mock_text_encoder, tokens)
        assert result.shape == (2, 77, 768)
        # Verify deterministic output
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.5))

    def test_encode_with_clip_skip(self, strategy_with_clip_skip, mock_text_encoder):
        """Test encoding with CLIP skip."""
        tokens = torch.arange(154).reshape(2, 77)
        result = strategy_with_clip_skip._encode_with_clip_skip(mock_text_encoder, tokens)
        assert result.shape == (2, 77, 768)
        # With clip_skip=2, should use second-to-last hidden state (0.5 * 0.9 = 0.45)
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.45))

    # Test _apply_weights_single_chunk
    def test_apply_weights_single_chunk(self, strategy):
        """Test applying weights for single chunk case."""
        encoder_hidden_states = torch.ones(2, 77, 768)
        weights = torch.ones(2, 1, 77) * 0.5
        result = strategy._apply_weights_single_chunk(encoder_hidden_states, weights)
        assert result.shape == (2, 77, 768)
        # Verify weights were applied: 1.0 * 0.5 = 0.5
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.5))

    # Test _apply_weights_multi_chunk
    def test_apply_weights_multi_chunk(self, strategy):
        """Test applying weights for multi-chunk case."""
        # Simulating 2 chunks: 2*75+2 = 152 tokens
        encoder_hidden_states = torch.ones(2, 152, 768)
        weights = torch.ones(2, 2, 77) * 0.5
        result = strategy._apply_weights_multi_chunk(encoder_hidden_states, weights)
        assert result.shape == (2, 152, 768)
        # Check that weights were applied to middle sections
        assert torch.allclose(result[0, 1, 0], torch.tensor(0.5))
        assert torch.allclose(result[0, 76, 0], torch.tensor(0.5))

    # Integration tests
    def test_encode_tokens_basic(self, strategy, mock_tokenize_strategy, mock_text_encoder):
        """Test basic token encoding flow."""
        tokens = torch.arange(154).reshape(2, 1, 77)
        models = [mock_text_encoder]
        tokens_list = [tokens]

        result = strategy.encode_tokens(mock_tokenize_strategy, models, tokens_list)

        assert len(result) == 1
        assert result[0].shape[0] == 2  # batch size
        assert result[0].shape[2] == 768  # hidden size
        # Verify deterministic output
        assert torch.allclose(result[0][0, 0, 0], torch.tensor(0.5))

    def test_encode_tokens_with_weights_single_chunk(self, strategy, mock_tokenize_strategy, mock_text_encoder):
        """Test weighted encoding with single chunk."""
        tokens = torch.arange(154).reshape(2, 1, 77)
        weights = torch.ones(2, 1, 77) * 0.5
        models = [mock_text_encoder]
        tokens_list = [tokens]
        weights_list = [weights]

        result = strategy.encode_tokens_with_weights(mock_tokenize_strategy, models, tokens_list, weights_list)

        assert len(result) == 1
        assert result[0].shape[0] == 2
        assert result[0].shape[2] == 768
        # Verify weights were applied: 0.5 (encoder output) * 0.5 (weight) = 0.25
        assert torch.allclose(result[0][0, 0, 0], torch.tensor(0.25))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
