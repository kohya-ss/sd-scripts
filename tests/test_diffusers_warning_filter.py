"""library.utils installs a filter that drops the spurious "should be kept in float32: []" warning of diffusers >= 0.40."""

import logging

import torch
from diffusers import AutoencoderKL

import library.utils  # noqa: F401 - installs the filter at import time


def _tiny_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=16,
    )


def _messages_during(fn) -> list:
    records = []
    diffusers_logger = logging.getLogger("diffusers.models.modeling_utils")

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = Collect()
    diffusers_logger.addHandler(h)
    try:
        fn()
    finally:
        diffusers_logger.removeHandler(h)
    return records


def test_spurious_fp32_warning_is_dropped():
    vae = _tiny_vae()
    msgs = _messages_during(lambda: vae.to(torch.float16))
    assert not any("should be kept in float32" in m for m in msgs), msgs


def test_real_fp32_warning_is_kept():
    vae = _tiny_vae()
    vae._keep_in_fp32_modules = ["quant_conv"]
    msgs = _messages_during(lambda: vae.to(torch.float16))
    assert any("should be kept in float32: ['quant_conv']" in m for m in msgs), msgs
