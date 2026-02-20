import pytest
import torch
from unittest.mock import Mock, patch

from library.strategy_sd import SdTokenizeStrategy


class TestSdTokenizeStrategy:
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock CLIP tokenizer."""
        tokenizer = Mock()
        tokenizer.model_max_length = 77
        tokenizer.bos_token_id = 49406
        tokenizer.eos_token_id = 49407
        tokenizer.pad_token_id = 49407

        def tokenize_side_effect(text, **kwargs):
            # Simple mock: return incrementing IDs based on text length
            # Real tokenizer would split into subwords
            num_tokens = min(len(text.split()), 75)
            input_ids = torch.arange(1, num_tokens + 1)

            if kwargs.get("return_tensors") == "pt":
                max_length = kwargs.get("max_length", 77)
                padded = torch.cat(
                    [
                        torch.tensor([tokenizer.bos_token_id]),
                        input_ids,
                        torch.tensor([tokenizer.eos_token_id]),
                        torch.full((max_length - num_tokens - 2,), tokenizer.pad_token_id),
                    ]
                )
                return Mock(input_ids=padded.unsqueeze(0))
            else:
                return Mock(
                    input_ids=torch.cat([torch.tensor([tokenizer.bos_token_id]), input_ids, torch.tensor([tokenizer.eos_token_id])])
                )

        tokenizer.side_effect = tokenize_side_effect
        return tokenizer

    @pytest.fixture
    def strategy_v1(self, mock_tokenizer):
        """Create a v1 strategy instance with mocked tokenizer."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=75, tokenizer_cache_dir=None)
            return strategy

    @pytest.fixture
    def strategy_v2(self, mock_tokenizer):
        """Create a v2 strategy instance with mocked tokenizer."""
        mock_tokenizer.pad_token_id = 0  # v2 has different pad token
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=True, max_length=75, tokenizer_cache_dir=None)
            return strategy

    # Test _split_on_break
    def test_split_on_break_no_break(self, strategy_v1):
        """Test splitting when no BREAK is present."""
        text = "a cat and a dog"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 1
        assert result[0] == "a cat and a dog"

    def test_split_on_break_single_break(self, strategy_v1):
        """Test splitting with single BREAK."""
        text = "a cat BREAK a dog"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 2
        assert result[0] == "a cat"
        assert result[1] == "a dog"

    def test_split_on_break_multiple_breaks(self, strategy_v1):
        """Test splitting with multiple BREAKs."""
        text = "a cat BREAK a dog BREAK a bird"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 3
        assert result[0] == "a cat"
        assert result[1] == "a dog"
        assert result[2] == "a bird"

    def test_split_on_break_case_sensitive(self, strategy_v1):
        """Test that BREAK splitting is case-sensitive."""
        text = "a cat break a dog"  # lowercase 'break' should not split
        result = strategy_v1._split_on_break(text)
        assert len(result) == 1
        assert result[0] == "a cat break a dog"

        text = "a cat Break a dog"  # mixed case should not split
        result = strategy_v1._split_on_break(text)
        assert len(result) == 1

    def test_split_on_break_with_whitespace(self, strategy_v1):
        """Test splitting with extra whitespace around BREAK."""
        text = "a cat  BREAK  a dog"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 2
        assert result[0] == "a cat"
        assert result[1] == "a dog"

    def test_split_on_break_empty_segments(self, strategy_v1):
        """Test splitting filters out empty segments."""
        text = "BREAK a cat BREAK BREAK a dog BREAK"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 2
        assert result[0] == "a cat"
        assert result[1] == "a dog"

    def test_split_on_break_only_break(self, strategy_v1):
        """Test splitting with only BREAK returns empty string."""
        text = "BREAK"
        result = strategy_v1._split_on_break(text)
        assert len(result) == 1
        assert result[0] == ""

    def test_split_on_break_empty_string(self, strategy_v1):
        """Test splitting empty string."""
        text = ""
        result = strategy_v1._split_on_break(text)
        assert len(result) == 1
        assert result[0] == ""

    # Test tokenize without BREAK
    def test_tokenize_single_text_no_break(self, strategy_v1):
        """Test tokenizing single text without BREAK."""
        text = "a cat"
        result = strategy_v1.tokenize(text)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].dim() == 3  # [batch, n_chunks, seq_len]

    def test_tokenize_list_no_break(self, strategy_v1):
        """Test tokenizing list of texts without BREAK."""
        texts = ["a cat", "a dog"]
        result = strategy_v1.tokenize(texts)
        assert len(result) == 1
        assert result[0].shape[0] == 2  # batch size

    # Test tokenize with BREAK
    def test_tokenize_single_break(self, strategy_v1):
        """Test tokenizing text with single BREAK."""
        text = "a cat BREAK a dog"
        result = strategy_v1.tokenize(text)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        # Should have concatenated tokens from both segments

    def test_tokenize_multiple_breaks(self, strategy_v1):
        """Test tokenizing text with multiple BREAKs."""
        text = "a cat BREAK a dog BREAK a bird"
        result = strategy_v1.tokenize(text)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    def test_tokenize_list_with_breaks(self, strategy_v1):
        """Test tokenizing list where some texts have BREAKs."""
        texts = ["a cat BREAK a dog", "a bird"]
        result = strategy_v1.tokenize(texts)
        assert len(result) == 1
        assert result[0].shape[0] == 2  # batch size

    # Test tokenize_with_weights without BREAK
    def test_tokenize_with_weights_no_break(self, strategy_v1):
        """Test weighted tokenization without BREAK."""
        text = "a cat"
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert isinstance(tokens_list[0], torch.Tensor)
        assert isinstance(weights_list[0], torch.Tensor)
        assert tokens_list[0].shape == weights_list[0].shape

    def test_tokenize_with_weights_list_no_break(self, strategy_v1):
        """Test weighted tokenization of list without BREAK."""
        texts = ["a cat", "a dog"]
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(texts)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert tokens_list[0].shape[0] == 2  # batch size
        assert tokens_list[0].shape == weights_list[0].shape

    # Test tokenize_with_weights with BREAK
    def test_tokenize_with_weights_single_break(self, strategy_v1):
        """Test weighted tokenization with single BREAK."""
        text = "a cat BREAK a dog"
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert isinstance(tokens_list[0], torch.Tensor)
        assert isinstance(weights_list[0], torch.Tensor)
        assert tokens_list[0].shape == weights_list[0].shape

    def test_tokenize_with_weights_multiple_breaks(self, strategy_v1):
        """Test weighted tokenization with multiple BREAKs."""
        text = "a cat BREAK a dog BREAK a bird"
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert tokens_list[0].shape == weights_list[0].shape

    def test_tokenize_with_weights_list_with_breaks(self, strategy_v1):
        """Test weighted tokenization of list with BREAKs."""
        texts = ["a cat BREAK a dog", "a bird BREAK a fish"]
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(texts)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert tokens_list[0].shape[0] == 2  # batch size
        assert tokens_list[0].shape == weights_list[0].shape

    # Test weighted prompts (with attention syntax)
    def test_tokenize_with_weights_attention_syntax(self, strategy_v1):
        """Test weighted tokenization with attention syntax like (word:1.5)."""
        text = "a (cat:1.5) and a dog"
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        # Weights should differ from 1.0 for the emphasized word

    def test_tokenize_with_weights_attention_and_break(self, strategy_v1):
        """Test weighted tokenization with both attention syntax and BREAK."""
        text = "a (cat:1.5) BREAK a [dog:0.8]"
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        assert tokens_list[0].shape == weights_list[0].shape

    def test_break_splits_long_prompts_into_chunks(self, strategy_v1):
        """Test that BREAK causes long prompts to split into expected number of chunks."""
        # Create a prompt with 80 tokens before BREAK and 80 after
        # Each "word" typically becomes 1-2 tokens, so ~40-80 words for 80 tokens
        long_segment = " ".join([f"word{i}" for i in range(40)])  # ~80 tokens
        text = f"{long_segment} BREAK {long_segment}"
        
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        
        # With model_max_length=77, we expect:
        # - First segment: 80 tokens -> needs 2 chunks (77 + remainder)
        # - Second segment: 80 tokens -> needs 2 chunks (77 + remainder)
        # Total: 4 chunks (2 per segment)
        
        assert len(tokens_list) == 1
        assert len(weights_list) == 1
        
        # Check that we got multiple chunks by looking at the shape
        # The concatenated result should be longer than a single chunk (77 tokens)
        tokens = tokens_list[0]
        weights = weights_list[0]
        
        # Should have significantly more than 77 tokens due to concatenation
        assert tokens.shape[-1] > 77, f"Expected >77 tokens but got {tokens.shape[-1]}"
        
        # With 2 segments of ~80 tokens each, we expect ~160 total tokens after concatenation
        # (exact number depends on tokenizer behavior, but should be in this range)
        assert tokens.shape[-1] >= 150, f"Expected >=150 tokens for 2 long segments but got {tokens.shape[-1]}"
    
    def test_break_splits_result_in_proper_chunks(self, strategy_v1):
        """Test that BREAK splitting results in proper chunk structure."""
        # Segment 1: ~40 tokens, Segment 2: ~40 tokens
        segment1 = " ".join([f"word{i}" for i in range(20)])
        segment2 = " ".join([f"word{i}" for i in range(20, 40)])
        text = f"{segment1} BREAK {segment2}"
        
        tokens_list, weights_list = strategy_v1.tokenize_with_weights(text)
        
        tokens = tokens_list[0]
        weights = weights_list[0]
        
        # Should be concatenated from 2 segments
        # Each segment fits in one chunk (< 77 tokens), so total should be ~80 tokens
        assert tokens.shape == weights.shape
        assert tokens.shape[-1] > 40, "Should have tokens from both segments"

    # Test v1 vs v2
    def test_v1_vs_v2_initialization(self, mock_tokenizer):
        """Test that v1 and v2 are initialized differently."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy_v1 = SdTokenizeStrategy(v2=False, max_length=75)
            strategy_v2 = SdTokenizeStrategy(v2=True, max_length=75)

            assert strategy_v1.tokenizer is not None
            assert strategy_v2.tokenizer is not None
            assert strategy_v1.max_length == 77  # 75 + 2 for BOS/EOS
            assert strategy_v2.max_length == 77

    # Test max_length handling
    def test_max_length_none(self, mock_tokenizer):
        """Test that None max_length uses tokenizer's model_max_length."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=None)
            assert strategy.max_length == mock_tokenizer.model_max_length

    def test_max_length_custom(self, mock_tokenizer):
        """Test custom max_length."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=150)
            assert strategy.max_length == 152  # 150 + 2 for BOS/EOS


class TestEdgeCases:
    """Test edge cases for tokenization."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock CLIP tokenizer."""
        tokenizer = Mock()
        tokenizer.model_max_length = 77
        tokenizer.bos_token_id = 49406
        tokenizer.eos_token_id = 49407
        tokenizer.pad_token_id = 49407

        def tokenize_side_effect(text, **kwargs):
            num_tokens = min(len(text.split()), 75)
            input_ids = torch.arange(1, num_tokens + 1)

            if kwargs.get("return_tensors") == "pt":
                max_length = kwargs.get("max_length", 77)
                padded = torch.cat(
                    [
                        torch.tensor([tokenizer.bos_token_id]),
                        input_ids,
                        torch.tensor([tokenizer.eos_token_id]),
                        torch.full((max_length - num_tokens - 2,), tokenizer.pad_token_id),
                    ]
                )
                return Mock(input_ids=padded.unsqueeze(0))
            else:
                return Mock(
                    input_ids=torch.cat([torch.tensor([tokenizer.bos_token_id]), input_ids, torch.tensor([tokenizer.eos_token_id])])
                )

        tokenizer.side_effect = tokenize_side_effect
        return tokenizer

    def test_very_long_text_with_breaks(self, mock_tokenizer):
        """Test very long text with multiple BREAKs."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=75)
            # Create long text segments
            long_text = " ".join([f"word{i}" for i in range(50)])
            text = f"{long_text} BREAK {long_text} BREAK {long_text}"

            result = strategy.tokenize(text)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

    def test_break_at_boundaries(self, mock_tokenizer):
        """Test BREAK at start and end of text."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=75)

            # BREAK at start
            text = "BREAK a cat"
            result = strategy.tokenize(text)
            assert len(result) == 1

            # BREAK at end
            text = "a cat BREAK"
            result = strategy.tokenize(text)
            assert len(result) == 1

            # BREAK at both ends
            text = "BREAK a cat BREAK"
            result = strategy.tokenize(text)
            assert len(result) == 1

    def test_consecutive_breaks(self, mock_tokenizer):
        """Test multiple consecutive BREAKs."""
        with patch.object(SdTokenizeStrategy, "_load_tokenizer", return_value=mock_tokenizer):
            strategy = SdTokenizeStrategy(v2=False, max_length=75)
            text = "a cat BREAK BREAK BREAK a dog"
            result = strategy.tokenize(text)
            assert len(result) == 1
            # Should only create 2 segments (consecutive BREAKs create empty segments that are filtered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
