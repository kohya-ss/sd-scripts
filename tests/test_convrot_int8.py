"""Tests for ConvRot INT8 (library/convrot_int8_kernels.py, library/convrot_int8_utils.py) and its MOSS-Image wiring.

CPU tests exercise the eager fallback (exact math up to float rounding) and the streaming quantizer; CUDA tests exercise
the fused Triton kernels when available.
"""

import functools
import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from library import convrot_int8_kernels as kernels
from library import convrot_int8_utils as utils
from library import moss_image_utils
from library.safetensors_utils import WeightTransformHooks

GS = utils.CONVROT_GROUPSIZE  # 256
K, N, M = 512, 96, 64  # in_features (2 groups), out_features, tokens

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_triton = pytest.mark.skipif(not kernels.HAS_TRITON, reason="triton required")


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _payload(groupsize: int, **overrides) -> torch.Tensor:
    values = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    values.update(overrides)
    raw = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _triple(module: str = "linear", *, groupsize: int = 4, in_features: int = 16, out_features: int = 8):
    return {
        f"{module}.weight": torch.zeros(out_features, in_features, dtype=torch.int8),
        f"{module}.weight_scale": torch.ones(out_features, 1, dtype=torch.float32),
        f"{module}.comfy_quant": _payload(groupsize),
    }


def _load_prequantized(*paths, weight_hook=None, **kwargs):
    """Conversion-only load: an empty target list disables dynamic quantization."""
    quantizer = utils.ConvRotInt8Quantizer(target_layer_keys=[])
    state_dict = quantizer.load_and_quantize([str(path) for path in paths], None, weight_hook=weight_hook, **kwargs)
    return state_dict, quantizer


# ---------------------------------------------------------------------------
# kernels: rotation and quantization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [4, 16, 64, 256])
def test_hadamard_is_symmetric_orthogonal(size):
    h = kernels._build_hadamard(size, dtype=torch.float64)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h, torch.eye(size, dtype=torch.float64), atol=1e-12)


def test_hadamard_rejects_non_power_of_4():
    with pytest.raises(ValueError, match="power of 4"):
        kernels._build_hadamard(128)


def test_quantize_dequant_roundtrip_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(N, K) * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, GS)
    assert wq.dtype == torch.int8 and ws.dtype == torch.float32 and ws.shape == (N, 1)
    w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, GS)
    assert _relerr(w_deq, w) < 1e-2  # row-wise int8 noise for gaussian weights is ~7e-3


def test_quantize_weight_convrot_skips_indivisible_and_non_2d():
    assert utils.quantize_weight_convrot("x.weight", torch.randn(8, 300)) is None  # 300 % 256 != 0
    assert utils.quantize_weight_convrot("x.weight", torch.randn(300)) is None  # not 2D
    result = utils.quantize_weight_convrot("x.weight", torch.randn(8, K))
    assert result is not None
    wq, ws, gs = result
    assert wq.shape == (8, K) and ws.shape == (8, 1) and gs == GS


def test_quantize_weight_convrot_picks_largest_dividing_groupsize():
    assert utils.select_convrot_groupsize(2688, (256, 64)) == 64  # 2688 = 42 * 64
    assert utils.select_convrot_groupsize(3072, (256, 64)) == 256
    assert utils.select_convrot_groupsize(100, (256, 64)) is None
    _, _, gs = utils.quantize_weight_convrot("x.weight", torch.randn(4, 2688), allowed_groupsizes=(256, 64))
    assert gs == 64


def test_quantize_weight_convrot_row_chunking_is_exact(monkeypatch):
    torch.manual_seed(0)
    w = torch.randn(70, K) * 0.02
    wq_ref, ws_ref, _ = utils.quantize_weight_convrot("x.weight", w)
    monkeypatch.setattr(utils, "QUANTIZE_ROW_CHUNK", 16)
    wq, ws, _ = utils.quantize_weight_convrot("x.weight", w)
    assert torch.equal(wq, wq_ref) and torch.equal(ws, ws_ref)


# ---------------------------------------------------------------------------
# autograd function: eager fallback (CPU)
# ---------------------------------------------------------------------------


def _make_quantized_linear(device, dtype, seed=0, bias=False):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, GS)
    w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, GS).to(dtype)
    b = torch.randn(N, device=device, dtype=dtype) * 0.1 if bias else None
    return wq, ws, w_deq, b


@pytest.mark.parametrize("bias", [False, True])
def test_forward_eager_cpu_matches_dequant_reference(bias):
    wq, ws, w_deq, b = _make_quantized_linear("cpu", torch.float32, bias=bias)
    x = torch.randn(M, K)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")
    ref = F.linear(x, w_deq, b)
    assert _relerr(out, ref) < 1e-5


def test_backward_eager_cpu_matches_dequant_reference():
    wq, ws, w_deq, b = _make_quantized_linear("cpu", torch.float32, bias=True)
    b.requires_grad_(True)
    x = torch.randn(2, M, K, requires_grad=True)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")
    g = torch.randn_like(out)
    out.backward(g)
    x_ref = x.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    F.linear(x_ref, w_deq, b_ref).backward(g)
    assert _relerr(x.grad, x_ref.grad) < 1e-5
    assert _relerr(b.grad, b_ref.grad) < 1e-5


def test_forward_autocast_casts_fp32_input_like_f_linear():
    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.bfloat16)
    x = torch.randn(M, K)  # fp32 input, as after an fp32 modulation
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
        ref = F.linear(x, w_deq)
    assert out.dtype == torch.bfloat16 == ref.dtype
    assert _relerr(out, ref) < 2e-2


def test_int8_backward_on_cpu_raises_a_clear_cuda_error():
    wq, ws, _, _ = _make_quantized_linear("cpu", torch.float32)
    x = torch.randn(M, K, requires_grad=True)
    with pytest.raises(RuntimeError, match=r"int8.*CUDA"):
        utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "int8").sum().backward()


# ---------------------------------------------------------------------------
# autograd function: fused Triton kernels (CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
@pytest.mark.parametrize("bias", [False, True])
def test_forward_triton_matches_dequant_reference(bias):
    wq, ws, w_deq, b = _make_quantized_linear("cuda", torch.bfloat16, bias=bias)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")
    ref = F.linear(x, w_deq, b)
    assert out.dtype == torch.bfloat16
    assert _relerr(out, ref) < 2e-2  # dynamic per-row int8 activation quantization


@requires_cuda
@requires_triton
@pytest.mark.parametrize(("bwd_mode", "tol"), [("bf16", 1e-2), ("int8", 3e-2)])
def test_grad_x_cuda_matches_dequant_reference(bwd_mode, tol):
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    x = torch.randn(2, M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, bwd_mode)
    g = torch.randn_like(out)
    out.backward(g)
    x_ref = x.detach().clone().requires_grad_(True)
    F.linear(x_ref, w_deq).backward(g)
    assert _relerr(x.grad, x_ref.grad) < tol


@requires_cuda
@requires_triton
def test_triton_and_eager_forward_parity(monkeypatch):
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out_triton = utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    monkeypatch.setattr(utils, "HAS_TRITON", False)
    out_eager = utils.ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    assert _relerr(out_triton, out_eager) < 2e-2


# ---------------------------------------------------------------------------
# embedding patch
# ---------------------------------------------------------------------------


def test_embedding_patch_dequantizes_and_rotates_back():
    torch.manual_seed(0)
    emb = nn.Embedding(50, K)
    w = emb.weight.detach() * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, GS)
    w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, GS)

    holder = nn.ModuleDict({"emb": emb})  # the patch addresses modules by their (non-empty) path
    sd = {"emb.weight": wq, "emb.scale_weight": ws}
    utils.apply_convrot_int8_monkey_patch(holder, sd, groupsize_map={"emb": GS}, embedding_dtype=torch.bfloat16)
    holder.requires_grad_(False)
    holder.load_state_dict(sd, strict=True, assign=True)

    ids = torch.tensor([[1, 5, 49], [0, 0, 7]])
    out = emb(ids)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 3, K)
    assert _relerr(out, w_deq[ids]) < 1e-2
    assert _relerr(out, w[ids]) < 1e-2
    assert emb.weight.dtype == torch.int8 and emb.forward.__func__ is utils.convrot_int8_embedding_forward_patch


# ---------------------------------------------------------------------------
# quantizer: pre-quantized ComfyUI triples
# ---------------------------------------------------------------------------


def test_probe_detects_comfy_quant_tensors(tmp_path):
    prequantized = tmp_path / "artifact.safetensors"
    save_file(_triple(), str(prequantized))
    ordinary = tmp_path / "ordinary.safetensors"
    save_file({"linear.weight": torch.zeros(8, 16, dtype=torch.bfloat16)}, str(ordinary))

    assert utils.has_comfy_quant_tensors([prequantized]) is True
    assert utils.has_comfy_quant_tensors([ordinary]) is False
    assert utils.has_comfy_quant_tensors([ordinary, prequantized]) is True


def test_conversion_canonicalizes_comfy_scales_and_keeps_per_layer_groups(tmp_path):
    path = tmp_path / "artifact.safetensors"
    tensors = {**_triple("a", groupsize=4, in_features=16, out_features=8), **_triple("b", groupsize=16, in_features=64, out_features=12)}
    tensors["b.comfy_attention.config"] = torch.tensor(list(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8)
    tensors["other.weight"] = torch.ones(3, dtype=torch.bfloat16)
    save_file(tensors, str(path))

    state_dict, quantizer = _load_prequantized(path)

    assert state_dict["a.weight"].dtype is torch.int8
    assert state_dict["a.scale_weight"].dtype is torch.float32 and state_dict["a.scale_weight"].shape == (8, 1)
    assert "a.weight_scale" not in state_dict and "a.comfy_quant" not in state_dict
    assert "b.comfy_attention.config" not in state_dict  # ComfyUI runtime hint, not a weight
    assert state_dict["other.weight"].dtype is torch.bfloat16  # passthrough
    assert quantizer.module_groupsizes == {"a": 4, "b": 16}
    assert quantizer.prequantized_count == 2 and quantizer.optimized_count == 0


@pytest.mark.parametrize(
    ("control", "match"),
    [
        (torch.tensor(list(b"{"), dtype=torch.uint8), "JSON"),
        (torch.tensor(list(b"[]"), dtype=torch.uint8), "object"),
        (_payload(4, format="float8_e4m3fn"), "format"),
        (_payload(4, convrot=False), "convrot"),
        (_payload(128), "power of 4"),
    ],
)
def test_conversion_rejects_invalid_control_payloads(tmp_path, control, match):
    tensors = _triple()
    tensors["linear.comfy_quant"] = control
    path = tmp_path / "bad-control.safetensors"
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match=match):
        _load_prequantized(path)


@pytest.mark.parametrize("missing_key", ["linear.weight", "linear.weight_scale"])
def test_conversion_rejects_missing_siblings(tmp_path, missing_key):
    tensors = _triple()
    del tensors[missing_key]
    path = tmp_path / "missing.safetensors"
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match=r"linear is missing tensors"):
        _load_prequantized(path)


def test_conversion_rejects_int8_weight_without_a_declared_triple(tmp_path):
    path = tmp_path / "orphan.safetensors"
    save_file({"orphan.weight": torch.zeros(8, 16, dtype=torch.int8)}, str(path))
    with pytest.raises(ValueError, match="comfy_quant"):
        _load_prequantized(path)


def test_conversion_rejects_lora_merge_into_prequantized_weights(tmp_path):
    path = tmp_path / "artifact.safetensors"
    save_file(_triple(), str(path))
    with pytest.raises(ValueError, match="Cannot merge LoRA"):
        _load_prequantized(path, weight_hook=lambda key, value, keep_on_calc_device=False: value)


def test_key_filter_drops_tensors_and_their_specs(tmp_path):
    path = tmp_path / "vlm.safetensors"
    tensors = {**_triple("model.language_model.layers.0.mlp.up_proj"), **_triple("lm_head"), **_triple("model.visual.blocks.0.attn.qkv")}
    tensors["model.visual.patch_embed.weight"] = torch.zeros(4, 16, dtype=torch.bfloat16)
    save_file(tensors, str(path))

    skip = ("lm_head.", "model.visual.")
    state_dict, quantizer = _load_prequantized(path, key_filter=lambda key: not key.startswith(skip))

    assert set(state_dict) == {
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.scale_weight",
    }
    assert quantizer.module_groupsizes == {"model.language_model.layers.0.mlp.up_proj": 4}


# ---------------------------------------------------------------------------
# quantizer: dynamic quantization + monkey patch + meta load
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(40, K)
        self.a = nn.Linear(K, N, bias=False)
        self.norm = nn.LayerNorm(N)
        self.b = nn.Linear(N, 12, bias=True)  # N = 96 is not divisible by 256: left in bf16

    def forward(self, ids):
        return self.b(self.norm(self.a(self.embed(ids))))


def test_quantizer_patch_and_meta_load_state_dict_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _TinyModel()
    with torch.no_grad():
        model.embed.weight.mul_(0.02)
        model.a.weight.mul_(0.5)
    ids = torch.tensor([[1, 2, 3, 39]])
    ref = model(ids)
    path = tmp_path / "model.safetensors"
    save_file({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, str(path))

    quantizer = utils.ConvRotInt8Quantizer(target_layer_keys=["embed", "a.", "b."], exclude_layer_keys=["norm"])
    sd = quantizer.load_and_quantize([str(path)], None)
    assert sd["a.weight"].dtype is torch.int8 and sd["a.scale_weight"].shape == (N, 1)
    assert sd["embed.weight"].dtype is torch.int8 and sd["embed.scale_weight"].shape == (40, 1)
    assert sd["b.weight"].dtype is torch.bfloat16 and "b.scale_weight" not in sd  # in_features 96 not divisible
    assert quantizer.module_groupsizes == {"embed": GS, "a": GS}
    assert quantizer.optimized_count == 2

    with torch.device("meta"):
        loaded = _TinyModel()
    utils.apply_convrot_int8_monkey_patch(loaded, sd, groupsize_map=quantizer.module_groupsizes, embedding_dtype=torch.float32)
    loaded.requires_grad_(False)
    loaded.load_state_dict(sd, strict=True, assign=True)

    assert type(loaded.a) is nn.Linear and loaded.a.weight.dtype is torch.int8 and not loaded.a.weight.requires_grad
    assert loaded.a._convrot_groupsize == GS and loaded.a._convrot_bwd_mode == "bf16"
    assert loaded.embed.forward.__func__ is utils.convrot_int8_embedding_forward_patch
    assert loaded.is_convrot_int8 is True and loaded.convrot_int8_layer_count == 2
    assert utils.is_convrot_int8_model(loaded) and not utils.is_convrot_int8_model(model)

    out = loaded.float()(ids)  # `.float()` must not touch the int8 weights nor break the forward
    assert loaded.a.weight.dtype is torch.int8 and loaded.a.scale_weight.dtype is torch.float32
    assert _relerr(out, ref) < 5e-2


def test_apply_patch_rejects_a_scale_for_a_missing_module():
    model = _TinyModel()
    with pytest.raises(ValueError, match="missing module ghost"):
        utils.apply_convrot_int8_monkey_patch(model, {"ghost.scale_weight": torch.ones(8, 1)}, groupsize_map={})


def test_apply_patch_rejects_non_linear_modules():
    model = _TinyModel()
    with pytest.raises(ValueError, match="neither nn.Linear nor nn.Embedding"):
        utils.apply_convrot_int8_monkey_patch(model, {"norm.scale_weight": torch.ones(N, 1)}, groupsize_map={})


def test_monkey_patch_int8_bwd_requires_triton(monkeypatch):
    monkeypatch.setattr(utils, "HAS_TRITON", False)
    model = _TinyModel()
    with pytest.raises(ValueError, match="triton"):
        utils.apply_convrot_int8_monkey_patch(model, {"a.scale_weight": torch.ones(N, 1)}, bwd_mode="int8")


def test_nonreentrant_checkpoint_backward_through_patched_linear():
    from torch.utils.checkpoint import checkpoint

    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.float32)
    holder = nn.ModuleDict({"lin": nn.Linear(K, N, bias=False)})
    sd = {"lin.weight": wq, "lin.scale_weight": ws}
    utils.apply_convrot_int8_monkey_patch(holder, sd, groupsize_map={"lin": GS})
    holder.requires_grad_(False)
    holder.load_state_dict(sd, strict=True, assign=True)
    lin = holder["lin"]
    lora = nn.Linear(K, N, bias=False)

    def block(x):
        return lin(x) + lora(x)

    x = torch.randn(M, K, requires_grad=True)
    checkpoint(block, x, use_reentrant=False).sum().backward()
    x_ref = x.detach().clone().requires_grad_(True)
    (F.linear(x_ref, w_deq) + lora(x_ref)).sum().backward()
    assert _relerr(x.grad, x_ref.grad) < 1e-5


# ---------------------------------------------------------------------------
# MOSS-Image wiring
# ---------------------------------------------------------------------------


def test_split_fused_qkv_handles_int8_triple():
    dim = 8
    split = functools.partial(moss_image_utils.split_fused_qkv, dim=dim)
    keys, _ = split("layers.3.proj.qkv.weight_scale", None)
    assert keys == ["layers.3.proj.q.weight_scale", "layers.3.proj.k.weight_scale", "layers.3.proj.v.weight_scale"]
    keys, tensors = split("model.diffusion_model.layers.3.proj.qkv.weight", torch.arange(3 * dim * dim, dtype=torch.int8).reshape(3 * dim, dim))
    assert keys[0] == "layers.3.proj.q.weight" and all(t.shape == (dim, dim) and t.dtype is torch.int8 for t in tensors)
    assert torch.equal(tensors[2], torch.arange(3 * dim * dim, dtype=torch.int8).reshape(3 * dim, dim)[2 * dim :])
    keys, tensors = split("layers.3.proj.qkv.weight_scale", torch.arange(3 * dim, dtype=torch.float32).reshape(3 * dim, 1))
    assert tensors[1].squeeze(1).tolist() == list(range(dim, 2 * dim))
    spec = _payload(GS)
    keys, tensors = split("layers.3.proj.qkv.comfy_quant", spec)
    assert keys == ["layers.3.proj.q.comfy_quant", "layers.3.proj.k.comfy_quant", "layers.3.proj.v.comfy_quant"]
    assert all(torch.equal(t, spec) for t in tensors)
    assert split("layers.3.proj.o.weight_scale", None) == (None, None)
    assert split("text_embedder.blocks.0.qkv.weight", None) == (None, None)


def test_quantizer_splits_comfyui_int8_qkv_through_the_adapter(tmp_path):
    dim = 8
    path = tmp_path / "dit_int8.safetensors"
    save_file(
        {
            "model.diffusion_model.layers.0.proj.qkv.weight": torch.arange(3 * dim * dim, dtype=torch.int8).reshape(3 * dim, dim),
            "model.diffusion_model.layers.0.proj.qkv.weight_scale": torch.arange(1, 3 * dim + 1, dtype=torch.float32).reshape(3 * dim, 1),
            "model.diffusion_model.layers.0.proj.qkv.comfy_quant": _payload(4),
            "model.diffusion_model.layers.0.comfy_attention.config": torch.tensor(list(b"{}"), dtype=torch.uint8),
            "model.diffusion_model.layers.0.norm1.weight": torch.ones(dim, dtype=torch.bfloat16),
        },
        str(path),
    )
    hooks = WeightTransformHooks()
    hooks.rename_hook = moss_image_utils._strip_dit_prefix
    hooks.split_hook = functools.partial(moss_image_utils.split_fused_qkv, dim=dim)
    quantizer = utils.ConvRotInt8Quantizer(moss_image_utils.DIT_CONVROT_INT8_TARGET_KEYS, moss_image_utils.DIT_CONVROT_INT8_EXCLUDE_KEYS)
    sd = quantizer.load_and_quantize([str(path)], None, weight_transform_hooks=hooks)

    assert set(sd) == {
        "layers.0.proj.q.weight",
        "layers.0.proj.k.weight",
        "layers.0.proj.v.weight",
        "layers.0.proj.q.scale_weight",
        "layers.0.proj.k.scale_weight",
        "layers.0.proj.v.scale_weight",
        "layers.0.norm1.weight",
    }
    assert sd["layers.0.proj.k.weight"].dtype is torch.int8
    assert sd["layers.0.proj.v.scale_weight"].squeeze(1).tolist() == list(range(2 * dim + 1, 3 * dim + 1))
    assert sd["layers.0.norm1.weight"].dtype is torch.bfloat16  # "norm" excluded, 1D anyway
    assert quantizer.module_groupsizes == {f"layers.0.proj.{c}": 4 for c in "qkv"}
    assert quantizer.prequantized_count == 3


def test_dit_target_keys_cover_the_main_block_linears_only():
    quantizer = utils.ConvRotInt8Quantizer(moss_image_utils.DIT_CONVROT_INT8_TARGET_KEYS, moss_image_utils.DIT_CONVROT_INT8_EXCLUDE_KEYS)
    for key in ["layers.0.proj.q.weight", "layers.41.proj.ffn_up_gate.weight", "layers.5.proj.ffn_down.weight"]:
        assert quantizer.is_target_key(key)
    for key in [
        "layers.0.norm1.weight",
        "layers.0.q_norm.weight",
        "text_embedder.proj.weight",
        "text_embedder.blocks.0.qkv.weight",
        "shared_block_modulation.weight",
        "video_embedder.weight",
        "final_linear_video.weight",
        "time_embedding.0.weight",
        "rope.bands",
    ]:
        assert not quantizer.is_target_key(key)


def test_text_encoder_target_keys_match_the_comfyui_scope():
    quantizer = utils.ConvRotInt8Quantizer(
        moss_image_utils.TEXT_ENCODER_CONVROT_INT8_TARGET_KEYS, moss_image_utils.TEXT_ENCODER_CONVROT_INT8_EXCLUDE_KEYS
    )
    for key in [
        "embed_tokens.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attn.out_proj.weight",
        "layers.3.self_attn.q_proj.weight",
        "layers.3.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ]:
        assert quantizer.is_target_key(key)
    for key in [
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.3.self_attn.q_norm.weight",
        "norm.weight",
    ]:
        assert not quantizer.is_target_key(key)


def test_trainer_convrot_options_and_hooks():
    import moss_image_train_network

    parser = moss_image_train_network.setup_parser()
    args = parser.parse_args(["--pretrained_model_name_or_path", "dit.safetensors", "--text_encoder", "te.safetensors"])
    assert args.convrot_int8 is False and args.convrot_int8_text_encoder is False and args.convrot_int8_bwd == "bf16"
    args = parser.parse_args(
        ["--pretrained_model_name_or_path", "d", "--text_encoder", "t", "--convrot_int8", "--convrot_int8_text_encoder", "--convrot_int8_bwd", "int8"]
    )
    assert args.convrot_int8 and args.convrot_int8_text_encoder and args.convrot_int8_bwd == "int8"

    trainer = moss_image_train_network.MossImageNetworkTrainer()
    assert trainer.cast_unet(args) is False  # keeps the fp32 scale_weight buffers

    # the grad checkpointing workaround must skip the int8 token embedding
    te = _TinyModel()
    sd = {"embed.weight": torch.zeros(40, K, dtype=torch.int8), "embed.scale_weight": torch.ones(40, 1)}
    utils.apply_convrot_int8_monkey_patch(te, sd, groupsize_map={"embed": GS})
    te.requires_grad_(False)
    te.embed.load_state_dict({"weight": sd["embed.weight"], "scale_weight": sd["embed.scale_weight"]}, assign=True)
    trainer.prepare_text_encoder_grad_ckpt_workaround(0, te)
    assert not te.embed.weight.requires_grad and te.a.weight.requires_grad


def test_inference_parser_has_convrot_options(monkeypatch):
    import moss_image_minimal_inference

    monkeypatch.setattr(
        "sys.argv", ["x", "--dit", "d", "--vae", "v", "--text_encoder", "t", "--save_path", "s", "--prompt", "p", "--convrot_int8"]
    )
    args = moss_image_minimal_inference.parse_args()
    assert args.convrot_int8 and not args.convrot_int8_text_encoder


def test_moss_image_block_forward_with_int8_projections_matches_float():
    """Regression: the block must not cast its activations to the (int8) projection weight dtype."""
    from library import moss_image_models

    torch.manual_seed(0)
    cfg = dict(dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
    ref = moss_image_models.MossImage(**cfg, attn_mode="torch", split_attn=False).eval()
    model = moss_image_models.MossImage(**cfg, attn_mode="torch", split_attn=False).eval()
    model.load_state_dict(ref.state_dict())

    sd = model.state_dict()
    groupsize_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith("layers."):
            wq, ws = kernels.quantize_int8_convrot_weight(module.weight.detach(), 64)
            sd[name + ".weight"] = wq
            sd[name + ".scale_weight"] = ws
            groupsize_map[name] = 64
    assert len(groupsize_map) == 2 * 6
    utils.apply_convrot_int8_monkey_patch(model, sd, groupsize_map=groupsize_map)
    model.requires_grad_(False)
    model.load_state_dict(sd, strict=True, assign=True)

    x = torch.randn(2, 8, 4, 6)
    t = torch.tensor([300.0, 800.0])
    ctx = torch.randn(2, 5, 24)
    mask = torch.ones(2, 5, dtype=torch.int32)
    mask[1, 3:] = 0
    with torch.no_grad():
        out_ref = ref(x, t, ctx, mask)
        out = model(x, t, ctx, mask)
    assert out.dtype == out_ref.dtype
    assert _relerr(out, out_ref) < 0.1  # ~1 with the bug (activations rounded to int8)
