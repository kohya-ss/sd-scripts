from types import SimpleNamespace

import pytest
import torch

from library import maruo_global_config as maruoCfg
from library import sdxl_original_unet
from sdxl_train_network import resolve_fp16_safe_norms_mode


def test_fp16_safe_norms_legacy_flag_resolves_to_strict():
    args = SimpleNamespace(fp16_safe_norms=True, fp16_safe_norms_mode=None)

    assert resolve_fp16_safe_norms_mode(args) == "strict"


def test_fp16_safe_norms_default_and_explicit_modes():
    assert resolve_fp16_safe_norms_mode(
        SimpleNamespace(fp16_safe_norms=False, fp16_safe_norms_mode=None)
    ) == "off"
    assert resolve_fp16_safe_norms_mode(
        SimpleNamespace(fp16_safe_norms=False, fp16_safe_norms_mode="strict")
    ) == "strict"
    assert resolve_fp16_safe_norms_mode(
        SimpleNamespace(fp16_safe_norms=False, fp16_safe_norms_mode="native_accum")
    ) == "native_accum"


def test_fp16_safe_norms_explicit_mode_can_refine_legacy_flag():
    assert resolve_fp16_safe_norms_mode(
        SimpleNamespace(fp16_safe_norms=True, fp16_safe_norms_mode="native_accum")
    ) == "native_accum"


def test_fp16_safe_norms_rejects_legacy_flag_with_explicit_off():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_fp16_safe_norms_mode(
            SimpleNamespace(fp16_safe_norms=True, fp16_safe_norms_mode="off")
        )


def test_fp16_safe_norms_rejects_invalid_mode_loaded_from_config():
    with pytest.raises(ValueError, match="must be one of"):
        resolve_fp16_safe_norms_mode(
            SimpleNamespace(fp16_safe_norms=False, fp16_safe_norms_mode="native_acum")
        )


def test_native_accum_falls_back_to_strict_layer_norm_on_cpu():
    norm = torch.nn.LayerNorm(16, dtype=torch.float16)
    x = torch.randn((2, 7, 16), dtype=torch.float16)
    expected = sdxl_original_unet._strict_fp16_layer_norm(x, norm)
    old_mode = maruoCfg.fp16_safe_norms_mode
    old_logged = sdxl_original_unet._NATIVE_FP16_LAYER_NORM_FALLBACK_LOGGED
    try:
        maruoCfg.fp16_safe_norms_mode = "native_accum"
        sdxl_original_unet._NATIVE_FP16_LAYER_NORM_FALLBACK_LOGGED = False
        actual = sdxl_original_unet._fp16_safe_layer_norm(x, norm)
    finally:
        maruoCfg.fp16_safe_norms_mode = old_mode
        sdxl_original_unet._NATIVE_FP16_LAYER_NORM_FALLBACK_LOGGED = old_logged

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 2016, 640), (1, 504, 1280)])
@pytest.mark.parametrize("scale", [1.0, 32.0, 2048.0])
def test_native_accum_layer_norm_matches_strict_output_and_input_gradient(shape, scale):
    torch.manual_seed(1234)
    norm = torch.nn.LayerNorm(shape[-1], device="cuda", dtype=torch.float16).requires_grad_(False)
    source = (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).half()
    grad_output = torch.randn(shape, device="cuda", dtype=torch.float16)

    strict_input = source.detach().requires_grad_(True)
    strict_output = sdxl_original_unet._strict_fp16_layer_norm(strict_input, norm)
    strict_output.backward(grad_output)

    native_input = source.detach().requires_grad_(True)
    old_mode = maruoCfg.fp16_safe_norms_mode
    try:
        maruoCfg.fp16_safe_norms_mode = "native_accum"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            native_output = sdxl_original_unet._fp16_safe_layer_norm(native_input, norm)
        native_output.backward(grad_output)
    finally:
        maruoCfg.fp16_safe_norms_mode = old_mode

    assert native_output.dtype == torch.float16
    assert strict_input.grad is not None and native_input.grad is not None
    torch.testing.assert_close(native_output, strict_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native_input.grad, strict_input.grad, rtol=0.0, atol=0.0)
    assert torch.isfinite(native_output).all()
    assert torch.isfinite(native_input.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_fp16_layer_norm_keeps_statistics_in_fp32():
    x = torch.randn((1, 17, 640), device="cuda", dtype=torch.float16)
    weight = torch.ones((640,), device="cuda", dtype=torch.float16)
    bias = torch.zeros((640,), device="cuda", dtype=torch.float16)

    with torch.autocast(device_type="cuda", enabled=False):
        output, mean, rstd = torch.native_layer_norm(x, (640,), weight, bias, 1.0e-5)

    assert output.dtype == torch.float16
    assert mean.dtype == torch.float32
    assert rstd.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "compile") or not hasattr(torch, "compiler"),
    reason="CUDA and torch.compile are required",
)
def test_native_fp16_layer_norm_compiles_as_fullgraph():
    torch.manual_seed(4321)
    norm = torch.nn.LayerNorm(640, device="cuda", dtype=torch.float16).requires_grad_(False)
    x = torch.randn((1, 64, 640), device="cuda", dtype=torch.float16)

    def run(value):
        return sdxl_original_unet._native_fp16_layer_norm(value, norm)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        expected = run(x)
        actual = torch.compile(run, backend="eager", fullgraph=True)(x)

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dim,heads,tokens", [(640, 10, 128), (1280, 20, 64)])
def test_native_accum_matches_strict_sdpa_transformer_block(dim, heads, tokens):
    torch.manual_seed(2468)
    block = sdxl_original_unet.BasicTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=64,
        cross_attention_dim=2048,
    ).to(device="cuda", dtype=torch.float16)
    block.set_use_sdpa(True)
    block.requires_grad_(False)
    source = torch.randn((1, tokens, dim), device="cuda", dtype=torch.float16)
    context_source = torch.randn((1, 77, 2048), device="cuda", dtype=torch.float16)
    grad_output = torch.randn_like(source)
    old_enabled = maruoCfg.fp16_safe_norms
    old_mode = maruoCfg.fp16_safe_norms_mode
    try:
        maruoCfg.fp16_safe_norms = True
        maruoCfg.fp16_safe_norms_mode = "strict"
        strict_input = source.detach().requires_grad_(True)
        strict_context = context_source.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            strict_output = block(strict_input, context=strict_context)
        strict_output.backward(grad_output)

        maruoCfg.fp16_safe_norms_mode = "native_accum"
        native_input = source.detach().requires_grad_(True)
        native_context = context_source.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            native_output = block(native_input, context=native_context)
        native_output.backward(grad_output)
    finally:
        maruoCfg.fp16_safe_norms = old_enabled
        maruoCfg.fp16_safe_norms_mode = old_mode

    assert strict_input.grad is not None and native_input.grad is not None
    assert strict_context.grad is not None and native_context.grad is not None
    torch.testing.assert_close(native_output, strict_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native_input.grad, strict_input.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native_context.grad, strict_context.grad, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_accum_mode_keeps_group_norm_strict():
    torch.manual_seed(5678)
    norm = sdxl_original_unet.GroupNorm32(32, 320, eps=1.0e-6).to(
        device="cuda", dtype=torch.float16
    ).requires_grad_(False)
    x = torch.randn((1, 320, 24, 24), device="cuda", dtype=torch.float16)

    old_enabled = maruoCfg.fp16_safe_norms
    old_mode = maruoCfg.fp16_safe_norms_mode
    try:
        maruoCfg.fp16_safe_norms = True
        maruoCfg.fp16_safe_norms_mode = "strict"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            expected = norm(x)
        maruoCfg.fp16_safe_norms_mode = "native_accum"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = norm(x)
    finally:
        maruoCfg.fp16_safe_norms = old_enabled
        maruoCfg.fp16_safe_norms_mode = old_mode

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
