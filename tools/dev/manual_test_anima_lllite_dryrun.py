"""Pseudo dry-run for Anima ControlNet-LLLite (no real Anima weights, no real data).

Verifies, end-to-end on CPU:
  1. LLLite construction over a stub DiT that uses the real ``library.anima_models.Attention``
  2. ``apply_to`` monkey-patches q_proj/k_proj/v_proj on selected attentions
  3. ``set_cond_image`` distributes cond_emb to all LLLite modules
  4. wrapper.forward propagates cond and reaches each patched Linear
  5. backward gives grads to LLLite params, but not to DiT params
  6. save_lllite_model -> reload into a fresh LLLite -> state_dicts match

Run:
    python tests/manual_test_anima_lllite_dryrun.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.anima_models import Attention  # real Anima Attention
from networks.control_net_lllite_anima import (
    ControlNetLLLiteDiT,
    AnimaControlNetLLLiteWrapper,
    save_lllite_model,
    load_lllite_weights,
)


# ---------------------------------------------------------------------------
# Stub DiT: holds a few real Attention modules but its forward bypasses
# attention math and just calls patched q_proj/k_proj/v_proj directly so we
# can drive the LLLite path without needing AttentionParams / RoPE / mask.
# ---------------------------------------------------------------------------
class _StubBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 4, head_dim: int = 16):
        super().__init__()
        # Real Anima Attention (self + cross)
        self.self_attn = Attention(
            query_dim=query_dim, context_dim=None, n_heads=n_heads, head_dim=head_dim
        )
        self.cross_attn = Attention(
            query_dim=query_dim, context_dim=context_dim, n_heads=n_heads, head_dim=head_dim
        )


class _StubDiT(nn.Module):
    def __init__(self, num_blocks: int = 3, query_dim: int = 64, context_dim: int = 96):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.blocks = nn.ModuleList(
            [_StubBlock(query_dim, context_dim) for _ in range(num_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,                    # (B, C=query_dim, T=1, latH, latW) — VAE latent (/8)
        timesteps: torch.Tensor,            # unused
        context: torch.Tensor,              # (B, S_ctx, context_dim)
        **kwargs,                           # padding_mask / source_attention_mask / t5_* unused
    ) -> torch.Tensor:
        # Mimic Anima's patchify (/2): latent (B, C, 1, latH, latW) -> token grid (B, latH/2 * latW/2, C)
        b, c, t, lat_h, lat_w = x.shape
        assert t == 1
        assert lat_h % 2 == 0 and lat_w % 2 == 0, "stub patchify needs even latent HW"
        x4 = x.squeeze(2)                                     # (B, C, latH, latW)
        x4 = F.avg_pool2d(x4, 2)                              # (B, C, latH/2, latW/2)
        h = lat_h // 2
        w = lat_w // 2
        seq = x4.view(b, c, h * w).permute(0, 2, 1).contiguous()  # (B, h*w, C) — post-patchify tokens

        out = seq
        for block in self.blocks:
            # Drive each patched Linear so LLLite forwards execute.
            q = block.self_attn.q_proj(out)
            k = block.self_attn.k_proj(out)
            v = block.self_attn.v_proj(out)
            sa = (q + k + v) / 3.0
            # Mix back to query_dim with output_proj (not LLLite-target)
            out = out + block.self_attn.output_proj(sa)

            q2 = block.cross_attn.q_proj(out)
            k2 = block.cross_attn.k_proj(context)  # context shape branch
            # cross.v_proj also exercised but is not LLLite-targeted per design
            v2 = block.cross_attn.v_proj(context)
            ca = q2 + k2.mean(dim=1, keepdim=True) + v2.mean(dim=1, keepdim=True)
            out = out + block.cross_attn.output_proj(ca)

        # Pseudo-unpatchify back to latent shape: (B, h*w, C) -> (B, C, 1, h, w)
        # (Real Anima would unpatchify to latH/latW; we keep h/w for shape consistency in the test only.)
        out = out.permute(0, 2, 1).contiguous().view(b, c, 1, h, w)
        return out


def _state_dicts_equal(sd_a: dict, sd_b: dict) -> bool:
    if set(sd_a.keys()) != set(sd_b.keys()):
        print(f"  KEY DIFF: only_a={set(sd_a)-set(sd_b)} only_b={set(sd_b)-set(sd_a)}")
        return False
    for k in sd_a:
        if sd_a[k].shape != sd_b[k].shape:
            print(f"  SHAPE DIFF at {k}: {sd_a[k].shape} vs {sd_b[k].shape}")
            return False
        if not torch.allclose(sd_a[k].float(), sd_b[k].float(), atol=0, rtol=0):
            print(f"  VALUE DIFF at {k}: max abs diff = {(sd_a[k].float()-sd_b[k].float()).abs().max().item()}")
            return False
    return True


def main():
    torch.manual_seed(0)

    # --- Build stub DiT + LLLite ---
    num_blocks = 3
    query_dim = 64
    context_dim = 96
    dit = _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim)
    dit.requires_grad_(False)

    target_layers = "self_attn_qkv_cross_q"  # 4 LLLite modules per block -> 12 total
    lllite = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=64,
        target_layers=target_layers,
    )
    expected_n_modules = num_blocks * 4
    assert len(lllite.lllite_modules) == expected_n_modules, (
        f"expected {expected_n_modules}, got {len(lllite.lllite_modules)}"
    )
    print(f"[1] built LLLite: {len(lllite.lllite_modules)} modules over {num_blocks} blocks")

    # state_dict sanity
    sd0 = lllite.state_dict()
    assert any(k.startswith("conditioning1.") for k in sd0)
    assert any(k.startswith("lllite_modules.0.down.") for k in sd0)
    assert all("org_module" not in k for k in sd0)
    print(f"[2] state_dict OK ({len(sd0)} keys, no org_module)")

    # --- apply_to + wrapper ---
    lllite.apply_to()
    wrapper = AnimaControlNetLLLiteWrapper(dit, lllite)
    print("[3] apply_to + wrapper built")

    # --- forward / set_cond_image / backward ---
    # Use real Anima geometry: x is VAE latent (/8); cond_image is original image (= latent * 8).
    # Token grid after patchify (/2 inside DiT) = latent/2.
    B = 2
    lat_H, lat_W = 8, 8       # latent spatial (must be even for patchify)
    img_H, img_W = lat_H * 8, lat_W * 8  # = 64 x 64 original image
    S_ctx = 7
    x = torch.randn(B, query_dim, 1, lat_H, lat_W, requires_grad=False)
    t = torch.zeros(B)
    ctx = torch.randn(B, S_ctx, context_dim)
    cond_image = torch.randn(B, 3, img_H, img_W)

    # zero-init: with cond_emb set, up.weight=0 so output should equal the cond=None reference
    out_no_cond = wrapper(x, t, ctx)  # cond_image=None
    out_zero_init = wrapper(x, t, ctx, cond_image=cond_image)
    assert torch.allclose(out_no_cond, out_zero_init, atol=1e-6), (
        "zero-init: forward with cond should match forward without cond before training"
    )
    print(f"[4] zero-init equivalence OK (out shape={tuple(out_zero_init.shape)})")

    # Perturb LLLite to break zero-init (so cond actually moves the output)
    with torch.no_grad():
        for m in lllite.lllite_modules:
            m.up.weight.normal_(0, 0.01)
    out_perturbed = wrapper(x, t, ctx, cond_image=cond_image)
    assert not torch.allclose(out_no_cond, out_perturbed, atol=1e-6), (
        "after perturbing up.weight, cond path should change the output"
    )
    print(f"[5] cond image actually moves output after perturbation")

    # backward: grads only on LLLite, not on DiT
    lllite.train()
    out = wrapper(x, t, ctx, cond_image=cond_image)
    loss = out.float().pow(2).mean()
    loss.backward()

    lllite_grad_count = 0
    lllite_grad_nonzero = 0
    for n, p in lllite.named_parameters():
        if p.grad is not None:
            lllite_grad_count += 1
            if p.grad.abs().sum().item() > 0:
                lllite_grad_nonzero += 1
    print(f"[6] LLLite grads: {lllite_grad_count} params have grad, {lllite_grad_nonzero} non-zero")
    assert lllite_grad_count > 0 and lllite_grad_nonzero > 0, "LLLite did not receive grad"

    dit_grad_seen = False
    for n, p in dit.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            dit_grad_seen = True
            print(f"  UNEXPECTED grad on dit param: {n}")
    assert not dit_grad_seen, "DiT params should not receive grad"
    print("[7] DiT received no grad (frozen as expected)")

    # --- save / load round-trip ---
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "lllite.safetensors")
        save_lllite_model(
            ckpt,
            lllite,
            dtype=torch.float32,
            metadata={
                "modelspec.architecture": "anima-preview/control-net-lllite",
                "lllite.target_layers": target_layers,
                "lllite.cond_emb_dim": "32",
                "lllite.mlp_dim": "64",
            },
        )
        assert os.path.exists(ckpt) and os.path.getsize(ckpt) > 0
        size_kb = os.path.getsize(ckpt) / 1024
        print(f"[8] saved {ckpt} ({size_kb:.1f} KB)")

        # fresh DiT + fresh LLLite, then load
        dit2 = _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim)
        dit2.requires_grad_(False)
        lllite2 = ControlNetLLLiteDiT(
            dit2, cond_emb_dim=32, mlp_dim=64, target_layers=target_layers
        )
        # before load: weights differ
        sd_before = lllite2.state_dict()
        sd_orig = lllite.state_dict()
        # at least up.weight should differ (we trained one but not the other)
        any_diff_before = any(
            not torch.allclose(sd_before[k].float(), sd_orig[k].float(), atol=1e-9)
            for k in sd_before
            if "up.weight" in k
        )
        assert any_diff_before, "fresh LLLite unexpectedly matches the trained one"
        load_lllite_weights(lllite2, ckpt, strict=True)
        sd_after = lllite2.state_dict()
        assert _state_dicts_equal(sd_orig, sd_after)
        print("[9] load round-trip OK (state_dicts match exactly)")

    print("\nAll dry-run checks PASSED.")


if __name__ == "__main__":
    main()
