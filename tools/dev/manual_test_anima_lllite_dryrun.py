"""Pseudo dry-run for Anima ControlNet-LLLite (no real Anima weights, no real data).

Verifies, end-to-end on CPU:
  1. LLLite construction over a stub DiT that uses the real ``library.anima_models.Attention``
  2. ``apply_to`` monkey-patches q_proj/k_proj/v_proj on selected attentions
  3. ``set_cond_image`` distributes cond_emb to all LLLite modules
  4. wrapper.forward propagates cond and reaches each patched Linear
  5. backward gives grads to LLLite params, but not to DiT params
  6. save_lllite_model -> reload into a fresh LLLite -> state_dicts match
  7. cond_input_space="latent" (v2.1): latent stem / mask pyramid build, forward, grads,
     save-load round-trip and pixel<->latent weight mix-up detection

Run:
    python tools/dev/manual_test_anima_lllite_dryrun.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# repo root on sys.path (this file lives in tools/dev/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from library.anima_models import Attention  # real Anima Attention
from networks.control_net_lllite_anima import (
    LATENT_COND_CHANNELS,
    ControlNetLLLiteDiT,
    AnimaControlNetLLLiteWrapper,
    build_cond_tensors,
    build_uncond_ref_context,
    encode_reference_hidden_states,
    install_ref_context_dispatch,
    parse_ref_blocks,
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

    check_latent_cond()
    check_semantic_trunk()

    print("\nAll dry-run checks PASSED.")


class _StubVae:
    """Minimal stand-in for the Qwen-Image VAE: /8 downsample into LATENT_COND_CHANNELS."""

    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(pixels, 8)  # (B, 3, H/8, W/8)
        return pooled.repeat(1, 6, 1, 1)[:, :LATENT_COND_CHANNELS]


def check_latent_cond():
    """v2.1: cond_input_space='latent' の構築 / forward / backward / round-trip / 取り違え検出."""
    torch.manual_seed(1)

    num_blocks = 2
    query_dim = 64
    context_dim = 96
    target_layers = "self_attn_q_pre"

    B = 2
    lat_H, lat_W = 8, 8                      # VAE latent 解像度 (= x の HW)
    img_H, img_W = lat_H * 8, lat_W * 8      # 元画像解像度
    S_ctx = 5

    x = torch.randn(B, query_dim, 1, lat_H, lat_W)
    t = torch.zeros(B)
    ctx = torch.randn(B, S_ctx, context_dim)
    rgb = torch.randn(B, 3, img_H, img_W).clamp(-1, 1)
    mask = (torch.rand(B, 1, img_H, img_W) > 0.5).float()
    vae = _StubVae()

    for cond_in_channels in (3, 4):
        dit = _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim)
        dit.requires_grad_(False)
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=32,
            mlp_dim=64,
            target_layers=target_layers,
            cond_in_channels=cond_in_channels,
            inpaint_masked_input=(cond_in_channels == 4),
            cond_input_space="latent",
        )
        lllite.apply_to()
        wrapper = AnimaControlNetLLLiteWrapper(dit, lllite)

        cond_image, cond_mask = build_cond_tensors(
            rgb,
            mask if cond_in_channels == 4 else None,
            cond_input_space="latent",
            cond_in_channels=cond_in_channels,
            inpaint_masked_input=(cond_in_channels == 4),
            vae=vae,
        )
        assert cond_image.shape == (B, LATENT_COND_CHANNELS, lat_H, lat_W), cond_image.shape
        if cond_in_channels == 4:
            assert cond_mask is not None and cond_mask.shape == (B, 1, img_H, img_W)
        else:
            assert cond_mask is None

        # zero-init: cond ありでも cond なしと一致する
        out_no_cond = wrapper(x, t, ctx)
        out_zero_init = wrapper(x, t, ctx, cond_image=cond_image, cond_mask=cond_mask)
        assert torch.allclose(out_no_cond, out_zero_init, atol=1e-6), (
            f"latent (cin={cond_in_channels}) zero-init mismatch"
        )

        # perturb して cond が出力を動かすことを確認
        with torch.no_grad():
            for m in lllite.lllite_modules:
                m.up.weight.normal_(0, 0.01)
        out_perturbed = wrapper(x, t, ctx, cond_image=cond_image, cond_mask=cond_mask)
        assert not torch.allclose(out_no_cond, out_perturbed, atol=1e-6), (
            f"latent (cin={cond_in_channels}) cond path did not move the output"
        )

        # backward: latent stem (と mask pyramid) に勾配が流れる
        lllite.train()
        wrapper(x, t, ctx, cond_image=cond_image, cond_mask=cond_mask).float().pow(2).mean().backward()
        stem_grads = {
            n: p.grad for n, p in lllite.named_parameters()
            if n.startswith("conditioning1.lat_") or n.startswith("conditioning1.mask_")
        }
        assert stem_grads, "no latent stem params found"
        for n, g in stem_grads.items():
            assert g is not None and g.abs().sum().item() > 0, f"no grad on {n}"
        if cond_in_channels == 4:
            assert any(n.startswith("conditioning1.mask_conv") for n in stem_grads), (
                "mask pyramid params missing in inpaint mode"
            )
        assert not any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in dit.parameters()
        ), "DiT params should not receive grad"

        print(
            f"[L1] latent cond_in_channels={cond_in_channels}: build / zero-init / cond effect / "
            f"grads OK ({len(stem_grads)} stem params)"
        )

        # save / load round-trip
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "lllite_latent.safetensors")
            save_lllite_model(ckpt, lllite, dtype=torch.float32, metadata={
                "lllite.cond_input_space": "latent",
                "lllite.cond_in_channels": str(cond_in_channels),
            })
            dit_b = _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim)
            lllite_b = ControlNetLLLiteDiT(
                dit_b, cond_emb_dim=32, mlp_dim=64, target_layers=target_layers,
                cond_in_channels=cond_in_channels, cond_input_space="latent",
            )
            load_lllite_weights(lllite_b, ckpt, strict=True)
            assert _state_dicts_equal(lllite.state_dict(), lllite_b.state_dict())
            print(f"[L2] latent cond_in_channels={cond_in_channels}: save/load round-trip OK")

            # pixel モデルに latent 重みを読ませると明示的に失敗する
            dit_px = _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim)
            lllite_px = ControlNetLLLiteDiT(
                dit_px, cond_emb_dim=32, mlp_dim=64, target_layers=target_layers,
                cond_in_channels=cond_in_channels, cond_input_space="pixel",
            )
            try:
                load_lllite_weights(lllite_px, ckpt, strict=False)
                raise AssertionError("loading latent weights into a pixel model should fail")
            except RuntimeError as e:
                assert "cond input space mismatch" in str(e), str(e)

            # 逆方向: pixel 重みを latent モデルに読ませても失敗する
            ckpt_px = os.path.join(tmp, "lllite_pixel.safetensors")
            save_lllite_model(ckpt_px, lllite_px, dtype=torch.float32)
            lllite_lat_c = ControlNetLLLiteDiT(
                _StubDiT(num_blocks=num_blocks, query_dim=query_dim, context_dim=context_dim),
                cond_emb_dim=32, mlp_dim=64, target_layers=target_layers,
                cond_in_channels=cond_in_channels, cond_input_space="latent",
            )
            try:
                load_lllite_weights(lllite_lat_c, ckpt_px, strict=False)
                raise AssertionError("loading pixel weights into a latent model should fail")
            except RuntimeError as e:
                assert "cond input space mismatch" in str(e), str(e)
            print(f"[L3] latent cond_in_channels={cond_in_channels}: pixel/latent mix-up rejected")


def check_semantic_trunk():
    """v3: trunk='semantic' を小型の実 Anima でエンドツーエンド検査する.

    encode_reference_hidden_states / t_embedding_norm hook / wrapper.forward /
    zero-init 等価 / cond・t の効き / 勾配 / save-load round-trip をカバーする。
    """
    from library.anima_models import Anima

    torch.manual_seed(2)

    num_blocks = 4
    model_channels = 64
    context_dim = 96
    dit = Anima(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=4,
        crossattn_emb_channels=context_dim,
        pos_emb_cls="rope3d",
        use_llm_adapter=False,
        attn_mode="torch",
    )
    # 素の Anima は AdaLN ゲートと final layer が zero-init で、ブロックの寄与が
    # すべてゼロゲートされる (= q_proj への摂動が出力に届かない)。テストとして
    # 意味を持たせるため、zero-init のパラメータをランダム化する。
    with torch.no_grad():
        for p in dit.parameters():
            if p.abs().sum().item() == 0:
                p.normal_(0, 0.02)
    dit.requires_grad_(False)
    dit.eval()

    lllite = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=32,
        target_layers="self_attn_q",
        cond_dim=32,
        cond_resblocks=1,
        cond_input_space="latent",
        trunk="semantic",
        ref_block=None,  # -> num_blocks // 2
        ref_timestep=0.0,
    )
    assert lllite.ref_blocks == (num_blocks // 2,)
    assert lllite.model_dim == model_channels
    assert lllite._t_hook_handle is not None, "t hook should be registered on the real Anima"
    lllite.apply_to()
    wrapper = AnimaControlNetLLLiteWrapper(dit, lllite)

    B, lat_H, lat_W = 2, 8, 8
    tok_H, tok_W = lat_H // 2, lat_W // 2
    x = torch.randn(B, 16, 1, lat_H, lat_W)
    t = torch.full((B,), 0.5)
    ctx = torch.randn(B, 7, context_dim)
    padding_mask = torch.zeros(B, 1, lat_H, lat_W)
    cond_latent = torch.randn(B, 16, lat_H, lat_W)

    # encode_reference_hidden_states 単体: shape と決定性
    with torch.no_grad():
        h_ref = encode_reference_hidden_states(dit, cond_latent, lllite.ref_blocks, 0.0, padding_mask)
        h_ref2 = encode_reference_hidden_states(dit, cond_latent, lllite.ref_blocks, 0.0, padding_mask)
    assert h_ref.shape == (B, 1, 1, tok_H, tok_W, model_channels), h_ref.shape
    assert torch.allclose(h_ref, h_ref2), "reference forward should be deterministic"
    # int 指定 (single) では 5-dim が返り、tuple 指定の K=1 と一致する
    with torch.no_grad():
        h_ref_int = encode_reference_hidden_states(dit, cond_latent, lllite.ref_blocks[0], 0.0, padding_mask)
    assert h_ref_int.shape == (B, 1, tok_H, tok_W, model_channels), h_ref_int.shape
    assert torch.allclose(h_ref[:, 0], h_ref_int)
    print(f"[S1] encode_reference_hidden_states OK ({tuple(h_ref.shape)})")

    # zero-init: cond ありでも cond なしと一致 (up=0 が支配)
    with torch.no_grad():
        out_no_cond = wrapper(x, t, ctx, padding_mask=padding_mask)
        out_zero = wrapper(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    assert torch.allclose(out_no_cond, out_zero, atol=1e-5), "v3 zero-init equivalence failed"
    # hook で t_local が配られている
    assert lllite.lllite_modules[0].t_local is not None
    assert lllite.lllite_modules[0].t_local.shape == (B, 1, 32)
    print(f"[S2] zero-init equivalence + t hook OK (out shape={tuple(out_zero.shape)})")

    # up を摂動すると cond が効く / 別の cond で出力が変わる (意味特徴が流れている)
    with torch.no_grad():
        for m in lllite.lllite_modules:
            m.up.weight.normal_(0, 0.01)
        out_pert = wrapper(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
        out_pert_other = wrapper(
            x, t, ctx, cond_image=torch.randn_like(cond_latent), padding_mask=padding_mask
        )
    assert not torch.allclose(out_no_cond, out_pert, atol=1e-6), "cond path did not move the output"
    assert not torch.allclose(out_pert, out_pert_other, atol=1e-6), (
        "different cond latents should give different outputs"
    )
    print("[S3] cond latent actually drives the output after perturbation")

    # t-FiLM: t_proj を摂動すると timestep により t_local が変わる
    with torch.no_grad():
        lllite.conditioning1.t_proj.weight.normal_(0, 0.05)
        wrapper(x, torch.full((B,), 0.1), ctx, cond_image=cond_latent, padding_mask=padding_mask)
        tl1 = lllite.lllite_modules[0].t_local.clone()
        wrapper(x, torch.full((B,), 0.9), ctx, cond_image=cond_latent, padding_mask=padding_mask)
        tl2 = lllite.lllite_modules[0].t_local.clone()
    assert not torch.allclose(tl1, tl2), "t_local should depend on the timestep via t_proj"
    print("[S4] t-FiLM path (per-forward t_local via hook) OK")

    # 勾配: trunk (ln_in/proj_in/resblocks/proj/t_proj) と gate に流れ、DiT には流れない
    lllite.train()
    out = wrapper(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    out.float().pow(2).mean().backward()
    named = dict(lllite.named_parameters())
    for key in (
        "conditioning1.ln_in.weight",
        "conditioning1.proj_in.weight",
        "conditioning1.proj.weight",
        "conditioning1.t_proj.weight",
        "lllite_modules.0.gate.weight",
        "lllite_modules.0.down.weight",
        "depth_embeds",
    ):
        p = named[key]
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"no grad on {key}"
    assert not any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in dit.parameters()
    ), "DiT params should not receive grad"
    print("[S5] grads flow to trunk / gate / t_proj, DiT frozen")

    # save / load round-trip + gate 可視化 capture
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "lllite_v3.safetensors")
        save_lllite_model(ckpt, lllite, dtype=torch.float32, metadata={
            "lllite.version": "3",
            "lllite.trunk": "semantic",
            "lllite.cond_input_space": "latent",
            "lllite.ref_block": lllite.ref_blocks_str,
            "lllite.ref_timestep": str(lllite.ref_timestep),
        })
        dit_b = Anima(
            max_img_h=64, max_img_w=64, max_frames=1, in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1, model_channels=model_channels,
            num_blocks=num_blocks, num_heads=4, crossattn_emb_channels=context_dim,
            pos_emb_cls="rope3d", use_llm_adapter=False, attn_mode="torch",
        )
        dit_b.requires_grad_(False)
        lllite_b = ControlNetLLLiteDiT(
            dit_b, cond_emb_dim=32, mlp_dim=32, target_layers="self_attn_q",
            cond_dim=32, cond_resblocks=1,
            cond_input_space="latent", trunk="semantic", ref_block=lllite.ref_blocks,
        )
        load_lllite_weights(lllite_b, ckpt, strict=True)
        assert _state_dicts_equal(lllite.state_dict(), lllite_b.state_dict())
        print("[S6] v3 save/load round-trip OK")

    # gate capture: token grid 形状の gate マップが取れる
    lllite.eval()
    for m in lllite.lllite_modules:
        m.capture_gate = True
    with torch.no_grad():
        wrapper(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    g = lllite.lllite_modules[0].last_gate
    assert g is not None and g.shape == (B, tok_H * tok_W, 1), g.shape
    assert lllite.last_cond_hw == (tok_H, tok_W)
    assert (g >= 0).all() and (g <= 1).all()
    for m in lllite.lllite_modules:
        m.capture_gate = False
    print(f"[S7] gate capture OK (shape={tuple(g.shape)}, grid={lllite.last_cond_hw})")

    # gradient checkpointing との相互作用: 参照フォワード (no_grad) + checkpoint 再計算の中で
    # cond_emb / t_local を参照しても勾配が正しく流れる
    lllite.zero_grad(set_to_none=True)
    dit.enable_gradient_checkpointing()
    dit.train()  # Block.forward の checkpoint 分岐は self.training が条件
    lllite.train()
    out_gc = wrapper(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    out_gc.float().pow(2).mean().backward()
    named = dict(lllite.named_parameters())
    for key in (
        "conditioning1.proj_in.weight",
        "conditioning1.t_proj.weight",
        "lllite_modules.0.gate.weight",
        "lllite_modules.0.down.weight",
        "depth_embeds",
    ):
        p = named[key]
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"[gc] no grad on {key}"
    assert not any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in dit.parameters()
    ), "[gc] DiT params should not receive grad"
    dit.disable_gradient_checkpointing()
    dit.eval()
    print("[S8] gradient checkpointing + semantic trunk grads OK")

    # ------------------------------------------------------------------
    # dual (ref_block 2 個の concat trunk) を同じ実 Anima でエンドツーエンド検査
    # ------------------------------------------------------------------
    # 既存 lllite の forward 差し替えを元に戻してから dual を貼り直す
    for m in lllite.lllite_modules:
        m.org_module[0].forward = m.org_forward
    lllite_dual = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=32,
        target_layers="self_attn_q",
        cond_dim=32,
        cond_resblocks=1,
        cond_input_space="latent",
        trunk="semantic",
        ref_block="1,3",
        ref_timestep=0.0,
    )
    assert lllite_dual.ref_blocks == (1, 3)
    assert lllite_dual.conditioning1.num_ref_blocks == 2
    lllite_dual.apply_to()
    wrapper_dual = AnimaControlNetLLLiteWrapper(dit, lllite_dual)

    # 参照フォワード: (B, K=2, 1, tok, tok, D)。K=1 の single 結果と各スライスが一致する
    with torch.no_grad():
        h_dual = encode_reference_hidden_states(dit, cond_latent, lllite_dual.ref_blocks, 0.0, padding_mask)
        h_b1 = encode_reference_hidden_states(dit, cond_latent, 1, 0.0, padding_mask)
        h_b3 = encode_reference_hidden_states(dit, cond_latent, 3, 0.0, padding_mask)
    assert h_dual.shape == (B, 2, 1, tok_H, tok_W, model_channels), h_dual.shape
    assert torch.allclose(h_dual[:, 0], h_b1) and torch.allclose(h_dual[:, 1], h_b3), (
        "dual reference forward slices should match the single-block forwards"
    )
    print(f"[D1] dual encode_reference_hidden_states OK ({tuple(h_dual.shape)})")

    # zero-init 等価 + 摂動後に両方の ref block が出力に効く
    with torch.no_grad():
        out_nc = wrapper_dual(x, t, ctx, padding_mask=padding_mask)
        out_z = wrapper_dual(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    assert torch.allclose(out_nc, out_z, atol=1e-5), "dual zero-init equivalence failed"
    # up が zero-init のままだと cond 経路に勾配が流れないので摂動してから backward する
    with torch.no_grad():
        for m in lllite_dual.lllite_modules:
            m.up.weight.normal_(0, 0.01)
    lllite_dual.train()
    out = wrapper_dual(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    out.float().pow(2).mean().backward()
    named = dict(lllite_dual.named_parameters())
    for key in (
        "conditioning1.ln_in.0.weight",
        "conditioning1.ln_in.1.weight",
        "conditioning1.proj_in.weight",
        "lllite_modules.0.gate.weight",
    ):
        p = named[key]
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"[dual] no grad on {key}"
    assert named["conditioning1.proj_in.weight"].shape == (32, 2 * model_channels)
    print("[D2] dual zero-init equivalence + grads (per-block ln_in) OK")

    # save/load round-trip + single 重みとの取り違え検出
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_d = os.path.join(tmp, "lllite_v3_dual.safetensors")
        save_lllite_model(ckpt_d, lllite_dual, dtype=torch.float32, metadata={
            "lllite.version": "3",
            "lllite.trunk": "semantic",
            "lllite.cond_input_space": "latent",
            "lllite.ref_block": lllite_dual.ref_blocks_str,
            "lllite.ref_timestep": str(lllite_dual.ref_timestep),
        })
        lllite_dual_b = ControlNetLLLiteDiT(
            dit, cond_emb_dim=32, mlp_dim=32, target_layers="self_attn_q",
            cond_dim=32, cond_resblocks=1,
            cond_input_space="latent", trunk="semantic", ref_block=(1, 3),
        )
        load_lllite_weights(lllite_dual_b, ckpt_d, strict=True)
        assert _state_dicts_equal(lllite_dual.state_dict(), lllite_dual_b.state_dict())
        try:
            load_lllite_weights(lllite_b, ckpt_d, strict=False)  # single モデルへ dual 重み
            raise AssertionError("loading dual weights into a single model should fail")
        except RuntimeError as e:
            assert "ref block count mismatch" in str(e), str(e)
    print("[D3] dual save/load round-trip + single/dual mix-up rejected OK")

    # ------------------------------------------------------------------
    # ref_block=-1 (x_embedder 出力) を検査
    # ------------------------------------------------------------------
    assert parse_ref_blocks("-1,3") == (-1, 3)
    assert parse_ref_blocks(-1) == (-1,)
    # D2 の wrapper 呼び出しで cond_emb が残っている (摂動済み up が生きている) ので、
    # 素の参照フォワードにするため明示的に clear する
    lllite_dual.clear_cond_image()
    with torch.no_grad():
        h_m1 = encode_reference_hidden_states(dit, cond_latent, -1, 0.0, padding_mask)
        x_emb, _, _ = dit.prepare_embedded_sequence(
            cond_latent.unsqueeze(2), fps=None, padding_mask=padding_mask
        )
    assert h_m1.shape == (B, 1, tok_H, tok_W, model_channels), h_m1.shape
    assert torch.allclose(h_m1, x_emb), "-1 must return the x_embedder output (pre-block)"
    # -1 は context / ref_timestep に依存しない
    with torch.no_grad():
        h_m1_ctx = encode_reference_hidden_states(
            dit, cond_latent, -1, 0.7, padding_mask, context=torch.randn(B, 5, context_dim)
        )
    assert torch.allclose(h_m1, h_m1_ctx), "-1 must be independent of context and ref_timestep"
    # dual (-1, 1): 先頭が x_embedder 出力、2 番目が block 1 出力
    with torch.no_grad():
        h_dm = encode_reference_hidden_states(dit, cond_latent, "-1,1", 0.0, padding_mask)
    assert h_dm.shape == (B, 2, 1, tok_H, tok_W, model_channels), h_dm.shape
    assert torch.allclose(h_dm[:, 0], x_emb) and torch.allclose(h_dm[:, 1], h_b1)
    # trunk 構築 (K=2) + メタデータ round-trip
    lllite_m = ControlNetLLLiteDiT(
        dit, cond_emb_dim=32, mlp_dim=32, target_layers="self_attn_q",
        cond_dim=32, cond_resblocks=1,
        cond_input_space="latent", trunk="semantic", ref_block="-1,1",
    )
    assert lllite_m.ref_blocks == (-1, 1)
    assert lllite_m.ref_blocks_str == "-1,1"
    assert parse_ref_blocks(lllite_m.ref_blocks_str) == (-1, 1)
    with torch.no_grad():
        cx_m = lllite_m.conditioning1(h_dm)
    assert cx_m.shape == (B, tok_H * tok_W, 32)
    print("[M1] ref_block=-1 (x_embedder output) OK")

    # ------------------------------------------------------------------
    # ref_context (zero / uncond / caption) + CFG dispatch を検査
    # ------------------------------------------------------------------
    # 小型 LLM Adapter を後付けして uncond / caption 経路を通す
    from library.anima_models import LLMAdapter

    torch.manual_seed(3)
    dit.llm_adapter = LLMAdapter(
        source_dim=48, target_dim=context_dim, model_dim=context_dim, num_layers=1, self_attn=False
    )
    dit.use_llm_adapter = True
    dit.llm_adapter.requires_grad_(False)
    dit.llm_adapter.eval()

    # ctor 検証: stem + ref_context != zero は reject
    try:
        ControlNetLLLiteDiT(
            dit, cond_emb_dim=32, mlp_dim=32, target_layers="self_attn_q",
            cond_dim=32, cond_input_space="latent", trunk="stem", ref_context="uncond",
        )
        raise AssertionError("stem + ref_context='uncond' should be rejected")
    except ValueError as e:
        assert "ref_context" in str(e), str(e)

    # uncond context: shape / 決定性 / zero-context との差
    with torch.no_grad():
        u1 = build_uncond_ref_context(dit, cond_latent.device, cond_latent.dtype)
        u2 = build_uncond_ref_context(dit, cond_latent.device, cond_latent.dtype)
    assert u1.shape == (1, 1, context_dim), u1.shape
    assert torch.allclose(u1, u2), "uncond ref context should be deterministic"
    with torch.no_grad():
        h_zero = encode_reference_hidden_states(dit, cond_latent, 1, 0.0, padding_mask)
        h_unc = encode_reference_hidden_states(dit, cond_latent, 1, 0.0, padding_mask, context=u1)
    assert not torch.allclose(h_zero, h_unc, atol=1e-6), (
        "a non-zero context should change the reference hidden states (cross-attn active)"
    )
    print(f"[R1] build_uncond_ref_context OK (shape={tuple(u1.shape)})")

    # wrapper._build_ref_context: 各モードの返り値
    for m in lllite_dual.lllite_modules:
        m.org_module[0].forward = m.org_forward
    lllite_r = ControlNetLLLiteDiT(
        dit, cond_emb_dim=32, mlp_dim=32, target_layers="self_attn_q",
        cond_dim=32, cond_resblocks=1,
        cond_input_space="latent", trunk="semantic", ref_block=1, ref_context="caption",
    )
    lllite_r.apply_to()
    wrapper_r = AnimaControlNetLLLiteWrapper(dit, lllite_r)

    with torch.no_grad():
        # caption + 推論経路 (t5 なし): context をそのまま返す
        assert wrapper_r._build_ref_context(ctx, {}) is ctx
        # caption + 学習経路 (t5 あり): adapter を通し padding をゼロ潰し
        src = torch.randn(B, 6, 48)
        t5_ids = torch.ones(B, 3, dtype=torch.long)
        t5_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        rc = wrapper_r._build_ref_context(
            src,
            {
                "t5_input_ids": t5_ids,
                "t5_attn_mask": t5_mask,
                "source_attention_mask": torch.ones(B, 6, dtype=torch.long),
            },
        )
        assert rc.shape == (B, 3, context_dim), rc.shape
        assert (rc[~t5_mask.bool()] == 0).all(), "padding tokens must be zeroed"
        # uncond: 定数キャッシュ (2 回目は同一オブジェクト)
        lllite_r.ref_context = "uncond"
        rc_u1 = wrapper_r._build_ref_context(ctx, {})
        rc_u2 = wrapper_r._build_ref_context(ctx, {})
        assert rc_u1 is rc_u2 and torch.allclose(rc_u1, u1)
        # zero: None
        lllite_r.ref_context = "zero"
        assert wrapper_r._build_ref_context(ctx, {}) is None
        lllite_r.ref_context = "caption"
    print("[R2] wrapper._build_ref_context (zero/uncond/caption, train/infer paths) OK")

    # caption の zero-init 等価 (wrapper end-to-end) + ref context が h_ref に効く
    with torch.no_grad():
        out_nc_r = wrapper_r(x, t, ctx, padding_mask=padding_mask)
        out_z_r = wrapper_r(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
    assert torch.allclose(out_nc_r, out_z_r, atol=1e-5), "caption zero-init equivalence failed"
    with torch.no_grad():
        for m in lllite_r.lllite_modules:
            m.up.weight.normal_(0, 0.01)
        out_c1 = wrapper_r(x, t, ctx, cond_image=cond_latent, padding_mask=padding_mask)
        ctx2 = torch.randn_like(ctx)
        out_c2 = wrapper_r(x, t, ctx2, cond_image=cond_latent, padding_mask=padding_mask)
        # 同じ ctx2 でも ref_context='zero' なら h_ref が変わる = context が参照に効いている
        lllite_r.ref_context = "zero"
        out_c2_zero = wrapper_r(x, t, ctx2, cond_image=cond_latent, padding_mask=padding_mask)
        lllite_r.ref_context = "caption"
    assert not torch.allclose(out_c1, out_c2, atol=1e-6)
    assert not torch.allclose(out_c2, out_c2_zero, atol=1e-6), (
        "caption vs zero ref context should change the output via h_ref"
    )
    print("[R3] caption ref context drives h_ref (wrapper end-to-end) OK")

    # install_ref_context_dispatch: context 照合で cond_emb が切り替わる
    ctx_a = torch.randn(B, 7, context_dim)
    ctx_b = torch.randn(B, 7, context_dim)
    with torch.no_grad():
        h_a = encode_reference_hidden_states(dit, cond_latent, 1, 0.0, padding_mask, context=ctx_a)
        h_b = encode_reference_hidden_states(dit, cond_latent, 1, 0.0, padding_mask, context=ctx_b)
        cx_a = lllite_r.conditioning1(h_a)
        cx_b = lllite_r.conditioning1(h_b)
    handle = install_ref_context_dispatch(dit, lllite_r, [(ctx_a, h_a), (ctx_b, h_b)])
    try:
        with torch.no_grad():
            dit(x, t, ctx_a, padding_mask=padding_mask)
            got_a = lllite_r.lllite_modules[0].cond_emb
            assert got_a is not None and torch.allclose(got_a, cx_a)
            dit(x, t, ctx_b, padding_mask=padding_mask)
            got_b = lllite_r.lllite_modules[0].cond_emb
            assert torch.allclose(got_b, cx_b) and not torch.allclose(got_b, cx_a)
            # 値一致 (同値の別テンソル) でも切り替わる
            dit(x, t, ctx_a.clone(), padding_mask=padding_mask)
            assert torch.allclose(lllite_r.lllite_modules[0].cond_emb, cx_a)
            # 不一致はフォールバック (先頭 = positive)
            dit(x, t, torch.randn_like(ctx_a), padding_mask=padding_mask)
            assert torch.allclose(lllite_r.lllite_modules[0].cond_emb, cx_a)
    finally:
        handle.remove()
    # remove 後は切り替わらない
    lllite_r.clear_cond_image()
    with torch.no_grad():
        dit(x, t, ctx_b, padding_mask=padding_mask)
    assert lllite_r.lllite_modules[0].cond_emb is None
    print("[R4] install_ref_context_dispatch (per-CFG-branch h_ref switch) OK")


if __name__ == "__main__":
    main()
