"""
Regression tests for library.model_util.expand_unet_to_inpainting().

Covers:
  - Diffusers-style UNet (has .conv_in / register_to_config)
  - Custom SDXL UNet (has .input_blocks)
  - Weight layout: first 4 input channels copied, remaining 5 zeroed
  - Bias preserved when present, absent when source has no bias
  - dtype/device preserved on the new conv (regression test for commit
    9292224 — fp16 weights must stay fp16, target device must be honoured)
  - in_channels attribute updated to 9 on both shapes
  - Idempotent when called on an already-9ch UNet
  - Rejects unexpected channel counts
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.model_util import expand_unet_to_inpainting


# ---------------------------------------------------------------------------
# Fakes mirroring the two UNet shapes the function supports
# ---------------------------------------------------------------------------

class _FakeDiffusersUNet(nn.Module):
    """Mimics the surface used by expand_unet_to_inpainting on diffusers UNets."""

    def __init__(self, in_channels=4, out_channels=320, bias=True, dtype=torch.float32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, bias=bias).to(dtype=dtype)
        self.in_channels = in_channels
        self.config = {"in_channels": in_channels}

    def register_to_config(self, **kwargs):
        # diffusers' real implementation routes through _internal_dict; we just
        # need to verify it gets called with the new in_channels.
        self.config.update(kwargs)


class _FakeSDXLUNet(nn.Module):
    """Mimics the surface used on the custom SDXL UNet (input_blocks[0][0])."""

    def __init__(self, in_channels=4, out_channels=320, bias=True, dtype=torch.float32):
        super().__init__()
        first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=bias).to(dtype=dtype)
        # input_blocks[0] is itself a Sequential whose first child is the conv.
        self.input_blocks = nn.ModuleList([nn.Sequential(first_conv)])
        self.in_channels = in_channels


# ---------------------------------------------------------------------------
# Diffusers-style
# ---------------------------------------------------------------------------

def test_diffusers_expansion_replaces_conv_in():
    unet = _FakeDiffusersUNet(in_channels=4, out_channels=16)
    original_first4 = unet.conv_in.weight.detach().clone()
    original_bias = unet.conv_in.bias.detach().clone()

    expand_unet_to_inpainting(unet)

    assert unet.conv_in.in_channels == 9
    assert unet.conv_in.weight.shape == (16, 9, 3, 3)
    # First 4 channels preserved exactly
    assert torch.equal(unet.conv_in.weight[:, :4], original_first4)
    # Remaining 5 are zeroed
    assert torch.all(unet.conv_in.weight[:, 4:] == 0)
    # Bias is preserved as-is
    assert torch.equal(unet.conv_in.bias, original_bias)
    # Module-level attribute synced
    assert unet.in_channels == 9
    # Config synced through register_to_config
    assert unet.config["in_channels"] == 9


def test_diffusers_expansion_without_bias():
    unet = _FakeDiffusersUNet(bias=False)
    expand_unet_to_inpainting(unet)
    assert unet.conv_in.bias is None
    assert unet.conv_in.weight.shape[1] == 9


# ---------------------------------------------------------------------------
# Custom SDXL-style
# ---------------------------------------------------------------------------

def test_sdxl_expansion_replaces_first_input_block():
    unet = _FakeSDXLUNet(in_channels=4, out_channels=16)
    original_first4 = unet.input_blocks[0][0].weight.detach().clone()

    expand_unet_to_inpainting(unet)

    new_conv = unet.input_blocks[0][0]
    assert isinstance(new_conv, nn.Conv2d)
    assert new_conv.in_channels == 9
    assert new_conv.weight.shape == (16, 9, 3, 3)
    assert torch.equal(new_conv.weight[:, :4], original_first4)
    assert torch.all(new_conv.weight[:, 4:] == 0)
    assert unet.in_channels == 9


# ---------------------------------------------------------------------------
# dtype / device preservation (regression for commit 9292224)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expansion_preserves_dtype(dtype):
    unet = _FakeDiffusersUNet(dtype=dtype)
    expand_unet_to_inpainting(unet)
    assert unet.conv_in.weight.dtype == dtype
    if unet.conv_in.bias is not None:
        assert unet.conv_in.bias.dtype == dtype


def test_expansion_preserves_device_cpu():
    unet = _FakeDiffusersUNet().cpu()
    expand_unet_to_inpainting(unet)
    assert unet.conv_in.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expansion_preserves_device_cuda():
    unet = _FakeDiffusersUNet().cuda()
    expand_unet_to_inpainting(unet)
    assert unet.conv_in.weight.device.type == "cuda"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_already_9_channels_is_noop():
    unet = _FakeDiffusersUNet(in_channels=9)
    same_conv = unet.conv_in
    expand_unet_to_inpainting(unet)
    assert unet.conv_in is same_conv  # not replaced
    assert unet.conv_in.in_channels == 9


def test_unexpected_channel_count_raises():
    unet = _FakeDiffusersUNet(in_channels=8)
    with pytest.raises(ValueError, match="4 or 9"):
        expand_unet_to_inpainting(unet)


def test_unknown_unet_shape_raises():
    class Bare(nn.Module):
        pass

    with pytest.raises(AttributeError, match="unknown architecture"):
        expand_unet_to_inpainting(Bare())
