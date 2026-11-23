"""The AdaLN modulation layers of Anima must keep the autocast of the caller (bf16), so that a LoRA can be applied to them.

``torch.autocast(..., dtype=torch.float32, enabled=use_fp32)`` turned the autocast of the caller off when ``use_fp32`` is
False (bf16): the modulation layers ran with the dtypes as they are, and a LoRA with float32 weights failed with a dtype
mismatch on the bf16 input.
"""

import pytest
import torch
from torch import nn

from library import attention
from library.anima_models import Block, FinalLayer

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast is required")

DIM = 32


class _AddLoRA(nn.Module):
    """A minimal LoRA: adds the output of float32 down / up layers to the original Linear, like ``LoRAModule``."""

    def __init__(self, org: nn.Linear):
        super().__init__()
        self.org = org
        self.lora_down = nn.Linear(org.in_features, 4, bias=False)
        self.lora_up = nn.Linear(4, org.out_features, bias=False)

    def forward(self, x):
        return self.org(x) + self.lora_up(self.lora_down(x))


def _wrap_modulation_linears(module: nn.Module, device):
    """Replace every Linear of the ``adaln_modulation*`` Sequentials with a float32 LoRA wrapper. Returns the count."""
    count = 0
    for name, seq in module.named_children():
        if name.startswith("adaln_modulation"):
            for i, layer in enumerate(seq):
                if isinstance(layer, nn.Linear):
                    seq[i] = _AddLoRA(layer).to(device)  # the LoRA weights stay float32
                    count += 1
    return count


def _block_inputs(dtype, device):
    x = torch.randn(1, 1, 2, 2, DIM, device=device, dtype=dtype)
    emb = torch.randn(1, 1, DIM, device=device, dtype=dtype)
    context = torch.randn(1, 3, DIM, device=device, dtype=dtype)
    return x, emb, context


@cuda_only
@pytest.mark.parametrize("use_adaln_lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_block_modulation_accepts_float32_lora_under_autocast(dtype, use_adaln_lora):
    device = torch.device("cuda")
    block = Block(DIM, DIM, num_heads=2, use_adaln_lora=use_adaln_lora, adaln_lora_dim=8).to(device, dtype)
    assert _wrap_modulation_linears(block, device) == (6 if use_adaln_lora else 3)

    x, emb, context = _block_inputs(dtype, device)
    adaln_lora = torch.randn(1, 1, 3 * DIM, device=device, dtype=dtype) if use_adaln_lora else None
    attn_params = attention.AttentionParams.create_attention_params("torch", False)
    use_fp32 = dtype == torch.float16  # as in Anima.forward
    with torch.autocast("cuda", dtype=dtype):
        out = block(x, emb, context, attn_params, use_fp32, adaln_lora_B_T_3D=adaln_lora)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_final_layer_modulation_accepts_float32_lora_under_autocast(dtype):
    device = torch.device("cuda")
    final = FinalLayer(DIM, spatial_patch_size=2, temporal_patch_size=1, out_channels=4).to(device, dtype)
    assert _wrap_modulation_linears(final, device) == 1

    x, emb, _ = _block_inputs(dtype, device)
    with torch.autocast("cuda", dtype=dtype):
        out = final(x, emb, use_fp32=dtype == torch.float16)
    assert torch.isfinite(out).all()


@cuda_only
def test_modulation_runs_in_float32_for_fp16_and_in_bf16_for_bf16():
    """The dtype of the modulation output: float32 for fp16 (overflow protection, PR #2297), unchanged bf16 for bf16."""
    device = torch.device("cuda")
    seen = {}

    for dtype in (torch.float16, torch.bfloat16):
        block = Block(DIM, DIM, num_heads=2).to(device, dtype)
        block.adaln_modulation_mlp.register_forward_hook(lambda m, i, o, d=dtype: seen.__setitem__(d, o.dtype))
        x, emb, context = _block_inputs(dtype, device)
        attn_params = attention.AttentionParams.create_attention_params("torch", False)
        with torch.autocast("cuda", dtype=dtype):
            block(x, emb, context, attn_params, dtype == torch.float16)

    assert seen[torch.float16] == torch.float32
    assert seen[torch.bfloat16] == torch.bfloat16
