"""Tests for library/clip_text_model.py (CLIPTextModel layout compatibility wrapper).

The tests run on any transformers version: with < 5.6 the wrapper is not applied by
`wrap_clip_text_model` (the model already has `text_model`), so the wrapper class itself is
exercised by wrapping the inner transformer directly.
"""

import torch
from transformers import CLIPTextConfig, CLIPTextModel

from library.clip_text_model import (
    CLIPTextModelWrapper,
    is_flattened_clip_text_model,
    unwrap_clip_text_model,
    wrap_clip_text_model,
)

TINY_CONFIG = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=8,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-05,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    model_type="clip_text_model",
    projection_dim=16,
)

EXPECTED_KEYS = [
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
    "text_model.encoder.layers.0.self_attn.q_proj.weight",
    "text_model.encoder.layers.1.mlp.fc2.bias",
    "text_model.final_layer_norm.weight",
]


def _make_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return CLIPTextModel._from_config(CLIPTextConfig(**TINY_CONFIG))


def _force_wrap(model: CLIPTextModel) -> CLIPTextModelWrapper:
    # transformers >= 5.6: the model itself is flat; < 5.6: use the inner CLIPTextTransformer
    inner = model if is_flattened_clip_text_model(model) else model.text_model
    return CLIPTextModelWrapper(inner)


def test_wrap_keeps_legacy_state_dict_keys_and_module_names():
    model = wrap_clip_text_model(_make_model())
    keys = set(model.state_dict().keys())
    for k in EXPECTED_KEYS:
        assert k in keys, k
    assert all(k.startswith("text_model.") for k in keys)

    names = dict(model.named_modules())
    assert "text_model.encoder.layers.0.self_attn" in names
    assert "text_model.encoder.layers.0.mlp" in names
    assert names["text_model.encoder.layers.0.self_attn"].__class__.__name__ in ("CLIPAttention", "CLIPSdpaAttention")
    assert names["text_model.encoder.layers.0.mlp"].__class__.__name__ == "CLIPMLP"


def test_wrap_is_idempotent_and_unwrap_returns_transformers_model():
    model = wrap_clip_text_model(_make_model())
    assert wrap_clip_text_model(model) is model
    assert isinstance(unwrap_clip_text_model(model), CLIPTextModel)
    assert unwrap_clip_text_model(unwrap_clip_text_model(model)) is unwrap_clip_text_model(model)


def test_wrapper_forward_and_attributes_match_inner():
    model = _force_wrap(_make_model())
    inner = model.text_model
    input_ids = torch.tensor([[0, 5, 6, 7, 2, 1, 1, 1]])

    model.eval()
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)
        ref = inner(input_ids, output_hidden_states=True, return_dict=True)
    assert torch.equal(out.last_hidden_state, ref.last_hidden_state)
    assert torch.equal(out.pooler_output, ref.pooler_output)
    assert len(out.hidden_states) == TINY_CONFIG["num_hidden_layers"] + 1
    # the legacy attribute path used all over the training scripts
    assert torch.equal(model.text_model.final_layer_norm(out.hidden_states[-1]), out.last_hidden_state)

    assert model.config.hidden_size == TINY_CONFIG["hidden_size"]
    assert model.dtype == torch.float32
    assert model.device == next(inner.parameters()).device
    assert model.get_input_embeddings() is inner.embeddings.token_embedding
    assert model.state_dict().keys() == {"text_model." + k for k in inner.state_dict().keys()}


def test_wrapper_load_state_dict_with_legacy_keys():
    src = wrap_clip_text_model(_make_model())
    dst = _force_wrap(_make_model())
    legacy_sd = {k: v.clone() for k, v in src.state_dict().items()}
    info = dst.load_state_dict(legacy_sd, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    for k, v in legacy_sd.items():
        assert torch.equal(dst.state_dict()[k], v)


def test_wrapper_dtype_device_follow_module_to():
    model = _force_wrap(_make_model())
    model.to(dtype=torch.bfloat16)
    assert model.dtype == torch.bfloat16
    assert model.text_model.embeddings.token_embedding.weight.dtype == torch.bfloat16


def test_is_flattened_detection():
    raw = _make_model()
    # on any version: a wrapped model is never reported as flattened
    assert not is_flattened_clip_text_model(wrap_clip_text_model(raw))
    # < 5.6 has text_model, >= 5.6 does not: exactly one of raw / inner looks flat
    if hasattr(raw, "text_model"):
        assert not is_flattened_clip_text_model(raw)
        assert is_flattened_clip_text_model(raw.text_model)
    else:
        assert is_flattened_clip_text_model(raw)
