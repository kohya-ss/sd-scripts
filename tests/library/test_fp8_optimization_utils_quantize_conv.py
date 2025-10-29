import pytest
import torch

from library.fp8_optimization_utils import quantize_conv_weight

def test_quantize_conv_weight_tensor_mode():
    """Test tensor-wise quantization for conv weights."""
    weight = torch.randn(16, 3, 3, 3)  # out_channels, in_channels, kh, kw
    fp8_dtype = torch.float8_e4m3fn
    max_value = 448.0
    min_value = -448.0
    
    quantized, scale = quantize_conv_weight(
        "test_layer", weight, fp8_dtype, max_value, min_value, "tensor"
    )
    
    assert quantized.shape == weight.shape
    assert quantized.dtype == fp8_dtype
    assert scale.shape == (1,)


def test_quantize_conv_weight_channel_mode():
    """Test per-channel quantization for conv weights."""
    weight = torch.randn(16, 3, 3, 3)
    fp8_dtype = torch.float8_e4m3fn
    max_value = 448.0
    min_value = -448.0
    
    quantized, scale = quantize_conv_weight(
        "test_layer", weight, fp8_dtype, max_value, min_value, "channel"
    )
    
    assert quantized.shape == weight.shape
    assert quantized.dtype == fp8_dtype
    assert scale.shape == (16, 1)  # one scale per output channel


def test_quantize_conv_weight_block_mode():
    """Test block-wise quantization for conv weights."""
    weight = torch.randn(16, 8, 4, 4)  # spatial size = 8*4*4 = 128
    fp8_dtype = torch.float8_e4m3fn
    max_value = 448.0
    min_value = -448.0
    block_size = 64
    
    quantized, scale = quantize_conv_weight(
        "test_layer", weight, fp8_dtype, max_value, min_value, "block", block_size
    )
    
    assert quantized.shape == weight.shape
    assert quantized.dtype == fp8_dtype
    assert scale.shape == (16, 2, 1)  # 128 / 64 = 2 blocks per channel


def test_quantize_conv_weight_block_fallback():
    """Test block-wise fallback to channel mode when not divisible."""
    weight = torch.randn(8, 3, 3, 3)  # spatial size = 3*3*3 = 27, not divisible by 64
    fp8_dtype = torch.float8_e4m3fn
    max_value = 448.0
    min_value = -448.0
    block_size = 64
    
    quantized, scale = quantize_conv_weight(
        "test_layer", weight, fp8_dtype, max_value, min_value, "block", block_size
    )
    
    assert quantized.shape == weight.shape
    assert scale.shape == (8, 1)  # fallback to channel mode


def test_quantize_conv_weight_non_conv_tensor():
    """Test fallback for non-convolution tensors."""
    weight = torch.randn(128, 64)  # 2D tensor (e.g., linear layer)
    fp8_dtype = torch.float8_e4m3fn
    max_value = 448.0
    min_value = -448.0
    
    quantized, scale = quantize_conv_weight(
        "test_layer", weight, fp8_dtype, max_value, min_value, "channel"
    )
    
    assert quantized.shape == weight.shape
    assert scale.shape == (1,)  # should fallback to tensor mode


def test_quantize_conv_weight_invalid_mode():
    """Test that invalid quantization mode raises error."""
    weight = torch.randn(16, 3, 3, 3)
    fp8_dtype = torch.float8_e4m3fn
    
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        quantize_conv_weight(
            "test_layer", weight, fp8_dtype, 448.0, -448.0, "invalid_mode"
        )
