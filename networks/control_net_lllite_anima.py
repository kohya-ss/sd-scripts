import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Anima の対象クラス名 (library/anima_models.py)
TARGET_ATTENTION_CLASS = "Attention"
TARGET_MLP_CLASS = "GPT2FeedForward"

# LLM Adapter 配下は学習対象外
LLM_ADAPTER_NAME = "llm_adapter"

# state_dict メタデータに記録するアーキテクチャ世代
LLLITE_ARCH_VERSION = "2"

# cond 画像の入力空間 (v2.1 で追加。メタデータ欠落時は "pixel" = 旧重み互換)
COND_INPUT_SPACES: Tuple[str, ...] = ("pixel", "latent")

# latent モードで conditioning1 が受け取る VAE latent のチャネル数 (Qwen-Image VAE z_dim)
LATENT_COND_CHANNELS = 16


# ----------------------------------------------------------------------------
# target_layers: atomic specifiers と preset
# ----------------------------------------------------------------------------

# 各 atomic specifier は 1 種類の挿入位置 (= 入力摂動対象 Linear) を表す
ATOMIC_SPECIFIERS: Tuple[str, ...] = (
    "self_attn_q_pre",      # selfattn.q_proj
    "self_attn_kv_pre",     # selfattn.k_proj + v_proj (常にセット)
    "cross_attn_q_pre",     # crossattn.q_proj
    "mlp_fc1_pre",          # mlp.layer1 (GPT2FeedForward の fc1)
)

# 後方互換 + よく使う組合せの名前付き alias
PRESETS: dict = {
    "self_attn_q":            ("self_attn_q_pre",),
    "self_attn_qkv":          ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q":  ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> Tuple[str, ...]:
    """target_layers 指定文字列を canonical な atomic tuple に解決する.

    受理する形式:
      - preset 名 1 つ (例: "self_attn_qkv")
      - カンマ区切りの atomic specifier (例: "self_attn_q_pre,mlp_fc1_pre")

    返り値は ATOMIC_SPECIFIERS の順序にそろえた重複なしの tuple.
    """
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    # canonical 順序 + 重複除去
    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


def _gn(channels: int) -> nn.GroupNorm:
    """channels を割り切れる範囲で 8 を上限とする GroupNorm."""
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    """Pre-activation ResBlock: GN→SiLU→Conv3x3→GN→SiLU→Conv3x3 + skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


# ASPP デフォルト dilations (v2 設計ドキュメント §2 軸5 推奨)
ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    並列ブランチ:
      - dilation=1 のものは 1x1 conv (受容野=1)
      - dilation>1 は 3x3 conv with dilation
      - global average pool → 1x1 conv → bilinear upsample (resolution-agnostic)
    すべてを concat → 1x1 conv で元のチャネル数に戻す.
    """

    def __init__(self, ch: int, dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1  # + global
        self.proj = nn.Sequential(
            nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    """v2 conditioning trunk.

    input_space="pixel" (v2, 既定):
      in (B,C_in,H,W)                     # 元解像度の cond 画像
      -> Conv 4x4 s=4    + GN + SiLU      # cond_dim/2,  H/4
      -> Conv 3x3 s=1    + GN + SiLU      # cond_dim/2,  H/4   (受容野拡張)
      -> Conv 4x4 s=4    + GN + SiLU      # cond_dim,    H/16  (token 解像度)

    input_space="latent" (v2.1):
      in (B,16,h,w)                       # VAE encode 済みの正規化 latent, h=H/8
      [inpaint 時] mask (B,1,H,W) {-1,1}
        -> Conv 2x2 s=2 + SiLU            # 4,  H/2
        -> Conv 3x3 s=2 + SiLU            # 8,  H/4
        -> Conv 3x3 s=2 (linear)          # 16, h        を latent と concat -> (B,32,h,w)
      -> Conv 3x3 s=1 p=1 + GN + SiLU     # cond_dim,    h      (latent 解像度で局所混合)
      -> Conv 2x2 s=2     + GN + SiLU     # cond_dim,    h/2    (= token 解像度, DiT patchify /2 と整合)

    以降は両モード共通:
      -> ResBlock x N                     # cond_dim
      -> (ASPP)
      -> Conv 1x1                         # cond_emb_dim
      -> flatten (B, S, cond_emb_dim)
      -> LayerNorm

    cond_in_channels は常に **pixel 意味論** で指定する。デフォルト 3 (RGB のみ)、4 で
    inpainting (RGB+mask) 等。latent モードでの stem 実入力チャネル数 (16 / 32) は内部で導出する。
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        input_space: str = "pixel",
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        assert input_space in COND_INPUT_SPACES, (
            f"input_space must be one of {list(COND_INPUT_SPACES)}, got {input_space!r}"
        )
        ch_half = cond_dim // 2

        if input_space == "latent":
            # latent モードの cond_in_channels は pixel 意味論のまま (3=RGB, 4=RGB+mask)。
            # 実際の stem 入力チャネルは 16 (RGB のみ) / 32 (mask pyramid 併用) になる。
            assert cond_in_channels in (3, 4), (
                f"latent input space supports cond_in_channels 3 or 4, got {cond_in_channels}"
            )

        self.cond_in_channels = cond_in_channels
        self.input_space = input_space
        # latent モードの inpainting でのみ mask 用の conv pyramid を持つ
        # (pixel モードでは mask は cond_image の 4ch 目として conv1 に直接入る)
        self.use_mask_branch = input_space == "latent" and cond_in_channels == 4

        if input_space == "pixel":
            self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
            self.norm1 = _gn(ch_half)
            self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
            self.norm2 = _gn(ch_half)
            self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
            self.norm3 = _gn(cond_dim)
        else:
            if self.use_mask_branch:
                # mask (B,1,H,W) {-1,1} -> (B,16,h,w). 各段が 2x2 / 4x4 / 8x8 スケールを分担する。
                # 段間は SiLU のみ (GN は一様な mask 領域で「マスク面積・絶対レベル」の情報を
                # 消してしまうため置かない)、最終段は既存 proj と同じく linear。
                self.mask_conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0)
                self.mask_conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
                self.mask_conv3 = nn.Conv2d(8, LATENT_COND_CHANNELS, kernel_size=3, stride=2, padding=1)

            stem_in = LATENT_COND_CHANNELS * (2 if self.use_mask_branch else 1)
            self.lat_conv1 = nn.Conv2d(stem_in, cond_dim, kernel_size=3, stride=1, padding=1)
            self.lat_norm1 = _gn(cond_dim)
            self.lat_conv2 = nn.Conv2d(cond_dim, cond_dim, kernel_size=2, stride=2, padding=0)
            self.lat_norm2 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])

        # ASPP (オプション、ResBlock の後段に挿入)
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_space == "pixel":
            assert mask is None, (
                "pixel input space does not take a separate mask tensor "
                "(the mask is packed into cond_image as the 4th channel)"
            )
            h = F.silu(self.norm1(self.conv1(x)))
            h = F.silu(self.norm2(self.conv2(h)))
            h = F.silu(self.norm3(self.conv3(h)))
        else:
            if self.use_mask_branch:
                assert mask is not None, "latent inpainting mode requires a mask tensor"
                m = F.silu(self.mask_conv1(mask))
                m = F.silu(self.mask_conv2(m))
                m = self.mask_conv3(m)  # (B, 16, H/8, W/8)
                assert m.shape[-2:] == x.shape[-2:], (
                    f"mask pyramid output {tuple(m.shape[-2:])} does not match cond latent "
                    f"{tuple(x.shape[-2:])} (mask HW must be cond latent HW * 8)"
                )
                x = torch.cat([x, m], dim=1)  # (B, 32, h, w)
            else:
                assert mask is None, (
                    "mask tensor given but this LLLite is not in inpainting mode "
                    "(cond_in_channels must be 4)"
                )
            h = F.silu(self.lat_norm1(self.lat_conv1(x)))
            h = F.silu(self.lat_norm2(self.lat_conv2(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()  # (B, S, C)
        h = self.out_norm(h)
        return h


class LLLiteModuleDiT(nn.Module):
    """単一の Attention Linear (q_proj/k_proj/v_proj) に対し LLLite の補正 x + cx を注入する.

    v2: concat-then-mid をベースに FiLM (γ, β) を mid 出力に適用、SiLU 化、depth embedding 対応.
    """

    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # list 包みで nn.Module 登録を回避し、state_dict に元 Linear の重みが入らないようにする
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (γ, β), zero-init で identity (1+γ=1, β=0)
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # 親 ControlNetLLLiteDiT が set_cond_image で注入する。
        # cond_emb は全モジュールで共有される cx (B, S, cond_emb_dim)、
        # depth_emb はこのモジュール用の depth embedding (cond_emb_dim,)。
        # cx を共有参照することで N コピーを避け、加算は forward 内で行う。
        self.cond_emb: Optional[torch.Tensor] = None
        self.depth_emb: Optional[torch.Tensor] = None

        # 親 ControlNetLLLiteDiT が __init__ 末尾で layer_idx を設定する。
        # depth embedding の index 参照は set_cond_image 側で行う (torch.compile 対策)。
        self.layer_idx: int = -1

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力レイアウト:
        #   - self/cross attention の q/k/v: (B, S, D)         (Anima Block 内で flatten 済み)
        #   - mlp.layer1:                    (B, T, H, W, D)   (flatten されずに渡される)
        # ここでは後者を (B, T*H*W, D) に flatten して処理し、最後に元 shape へ復元する。
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        # cond_emb は全モジュール共有の cx、depth_emb はこのモジュール用の depth vector。
        # ここで加算することで cx の N コピーを避ける。整数 layer_idx は参照せず
        # テンソル属性 depth_emb だけを足すので、torch.compile は単一グラフを維持する
        # (gradient checkpointing 下では cond_local は領域内で再計算され retain されない)。
        cond_local = self.cond_emb + self.depth_emb  # (B, H*W, cond_emb_dim)

        # CFG 推論用 (学習時は通らない想定)
        if x.shape[0] // 2 == cond_local.shape[0]:
            cond_local = cond_local.repeat(2, 1, 1)

        # T=1 固定前提なので S == H*W のはず
        assert x.shape[1] == cond_local.shape[1], (
            f"LLLite seq mismatch ({self.lllite_name}): x={x.shape[1]} vs cond_emb={cond_local.shape[1]}"
        )

        h = F.silu(self.down(x))  # (B, S, mlp)

        # FiLM パラメータ (cond_local 由来、zero-init で identity)
        gb = self.cond_to_film(cond_local)  # (B, S, 2*mlp)
        gamma, beta = gb.chunk(2, dim=-1)

        mid_in = torch.cat([cond_local, h], dim=-1)  # (B, S, cond+mlp)
        m = self.mid(mid_in)
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        y = self.org_forward(x + out)  # (B, S, D_out)

        if is_5d:
            # org Linear の出力次元は in_features と異なりうるので最後だけ -1 で復元
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


class ControlNetLLLiteDiT(nn.Module):
    """Anima DiT 用の ControlNet-LLLite 本体. conditioning1 を共有保持し、各対象 Linear に LLLite を貼る.

    target_layers は preset 名または atomic specifier のカンマ区切りで指定する (parse_target_layers 参照).
    """

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
        cond_input_space: str = "pixel",
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)
        assert cond_input_space in COND_INPUT_SPACES, (
            f"cond_input_space must be one of {list(COND_INPUT_SPACES)}, got {cond_input_space!r}"
        )

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers          # ユーザ指定そのまま (記録用)
        self.target_atomics = atomics                # canonical atomic tuple
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        # 4ch (RGB+mask) inpainting 用の付加情報。inpaint_masked_input は学習側の RGB マスキング方針を
        # 記録するためのフラグで、モデル forward の挙動には影響しない (メタデータ復元用)。
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input
        # "pixel": cond image をそのまま stem に通す (v2)
        # "latent": VAE encode 済み latent を stem に通す (v2.1)。inpaint の mask は別テンソル
        self.cond_input_space = cond_input_space

        # pixel : cond image  (B, cond_in_channels, H*16, W*16) -> (B, S, cond_emb_dim)
        # latent: cond latent (B, 16, H*2, W*2) (+ mask (B,1,H*16,W*16)) -> (B, S, cond_emb_dim)
        self.conditioning1 = _Conditioning1(
            cond_dim, cond_emb_dim, cond_resblocks,
            use_aspp=use_aspp, aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
            input_space=cond_input_space,
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)

        # depth embedding: 各モジュール用の zero-init bias (N, cond_emb_dim)
        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        inpaint_info = (
            f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels == 4 else ""
        )
        logger.info(
            f"ControlNet-LLLite (Anima v{LLLITE_ARCH_VERSION}): created {n} modules for "
            f"target={target_layers!r} (atomics={list(atomics)}), "
            f"cond_input={cond_input_space}, "
            f"cond_in_channels={cond_in_channels}, cond_dim={cond_dim}, cond_resblocks={cond_resblocks}, {aspp_info}, "
            f"cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}{inpaint_info}"
        )

    @property
    def target_atomics_str(self) -> str:
        """canonical atomic specifier をカンマ区切り文字列で返す (メタデータ保存用)."""
        return ",".join(self.target_atomics)

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: Tuple[str, ...]) -> bool:
        # 常時スキップ
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            # cross_attn の K,V は text 側で shape 不一致なので非対応
            return False

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: Tuple[str, ...],
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            # LLM Adapter 配下は除外 (クラス名でほぼ落ちるが name でも明示防御)
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not isinstance(child, nn.Linear):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                # GPT2FeedForward.layer1 = fc1 (d_model -> d_ff)
                child = getattr(module, "layer1", None)
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor], cond_mask: Optional[torch.Tensor] = None):
        """cond_image を conditioning1 に通し、全 LLLite モジュールに cond_emb を配る。None で解除。

        pixel モード : cond_image = (B, cond_in_channels, H*16, W*16)、cond_mask は不可
        latent モード: cond_image = (B, 16, H*2, W*2) の正規化済み VAE latent、
                       inpaint (cond_in_channels=4) のとき cond_mask = (B, 1, H*16, W*16) in [-1, 1]
        """
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
                m.depth_emb = None
            return
        cx = self.conditioning1(cond_image, cond_mask)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            # 共有の cx を全モジュールに同一テンソルとして持たせ (N コピーを避ける)、
            # depth embedding はこのモジュール用の (cond_emb_dim,) スライスだけを渡す。
            # 加算は forward 内で行う (cx の N 倍メモリを回避するため)。
            #
            # depth_embeds[layer_idx] の index 参照はここ (compile 領域の外) で毎ステップ
            # 行う。forward 内で整数 layer_idx を参照すると torch.compile がブロック毎に
            # 別グラフを焼くため、参照はここに残す。また index 参照を毎ステップ行うことで
            # SelectBackward が毎回張り直され、2 回目以降の backward でも depth_embeds へ
            # 正しく勾配が流れる (__init__ で一度だけ index すると graph 再利用で破綻する)。
            m.cond_emb = cx  # 全モジュールで共有 (同一テンソル)
            m.depth_emb = self.depth_embeds[m.layer_idx]  # (cond_emb_dim,), broadcast over (B, S)

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()


class AnimaControlNetLLLiteWrapper(nn.Module):
    """accelerator.prepare に渡す最上位 nn.Module.
    forward 内で lllite.set_cond_image を呼んで cond の計算を accumulate/autocast/DDP スコープに入れる."""

    def __init__(self, dit: nn.Module, lllite: ControlNetLLLiteDiT):
        super().__init__()
        self.dit = dit
        self.lllite = lllite

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cond_image: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # T=1 固定
        assert x.shape[2] == 1, f"Anima LLLite supports T=1 only, got T={x.shape[2]}"
        if cond_image is not None:
            latent_h, latent_w = x.shape[-2], x.shape[-1]
            if self.lllite.cond_input_space == "pixel":
                # 解像度整合チェック: x は VAE latent (/8)、cond_image は元画像 (/1)。
                # patchify (/2) は DiT 内部 (prepare_embedded_sequence) で実施されるため、
                # ここでは latent HW * 8 == cond_image HW を期待する。
                # conditioning1 (stride 16) は cond_image を /16 = latent/2 = token 空間に揃える。
                expected_h = latent_h * 8
                expected_w = latent_w * 8
                assert cond_image.shape[-2] == expected_h and cond_image.shape[-1] == expected_w, (
                    f"cond_image HW mismatch: latent={latent_h}x{latent_w} -> expected "
                    f"{expected_h}x{expected_w}, got {cond_image.shape[-2]}x{cond_image.shape[-1]}"
                )
                expected_c = self.lllite.cond_in_channels
                assert cond_image.shape[1] == expected_c, (
                    f"cond_image channel mismatch: expected {expected_c} (cond_in_channels), "
                    f"got {cond_image.shape[1]}"
                )
                assert cond_mask is None, (
                    "cond_mask is only used with cond_input_space='latent'; in pixel mode the mask "
                    "must be packed into cond_image as the 4th channel"
                )
            else:
                # latent モード: cond_image は VAE encode 済み latent なので x と同一解像度
                assert cond_image.shape[-2] == latent_h and cond_image.shape[-1] == latent_w, (
                    f"cond latent HW mismatch: expected {latent_h}x{latent_w} (same as x), "
                    f"got {cond_image.shape[-2]}x{cond_image.shape[-1]}"
                )
                assert cond_image.shape[1] == LATENT_COND_CHANNELS, (
                    f"cond latent channel mismatch: expected {LATENT_COND_CHANNELS}, "
                    f"got {cond_image.shape[1]}"
                )
                if self.lllite.cond_in_channels == 4:
                    assert cond_mask is not None, (
                        "cond_mask is required for inpainting (cond_in_channels=4) in latent mode"
                    )
                    expected_h = latent_h * 8
                    expected_w = latent_w * 8
                    assert cond_mask.shape[1] == 1 and (
                        cond_mask.shape[-2] == expected_h and cond_mask.shape[-1] == expected_w
                    ), (
                        f"cond_mask shape mismatch: expected (B,1,{expected_h},{expected_w}), "
                        f"got {tuple(cond_mask.shape)}"
                    )
                else:
                    assert cond_mask is None, (
                        f"cond_mask given but cond_in_channels={self.lllite.cond_in_channels} "
                        "(mask is only used in inpainting mode)"
                    )
            self.lllite.set_cond_image(cond_image, cond_mask)
        return self.dit(x, timesteps, context, **kwargs)


# ---------------------------------------------------------------------------
# cond tensor 組み立て (学習ループ / sample hook / 推論スクリプトで共有)
# ---------------------------------------------------------------------------


def build_cond_tensors(
    rgb: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    cond_input_space: str = "pixel",
    cond_in_channels: int = 3,
    inpaint_masked_input: bool = False,
    vae=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """cond 画像 (と inpaint mask) から wrapper / set_cond_image に渡す組を作る。

    Args:
        rgb: (B, 3, H, W), [-1, 1] 正規化済み
        mask: (B, 1, H, W) in {0, 1} (1=inpaint 域)。cond_in_channels=4 のとき必須
        cond_input_space: "pixel" (v2) or "latent" (v2.1)
        cond_in_channels: pixel 意味論のチャネル数 (3=RGB, 4=RGB+mask)
        inpaint_masked_input: True なら mask 域の RGB を 0 に潰してから使う
        vae: latent モードで必須。encode_pixels_to_latents を持つ Qwen-Image VAE

    Returns:
        (cond_image, cond_mask)
          pixel : cond_image=(B,3 or 4,H,W)、cond_mask=None
          latent: cond_image=(B,16,H/8,W/8) の正規化済み latent、
                  cond_mask=(B,1,H,W) in [-1,1] (inpaint 時のみ、それ以外は None)

    mask のチャネル正規化 ({0,1} -> {-1,1}) はここで行う。両モードで規約を揃えるため、
    latent モードでも mask は pixel 解像度のまま返す (conditioning1 内の pyramid が /8 に落とす)。
    """
    assert cond_input_space in COND_INPUT_SPACES, (
        f"cond_input_space must be one of {list(COND_INPUT_SPACES)}, got {cond_input_space!r}"
    )
    is_inpaint = cond_in_channels == 4
    if is_inpaint:
        assert mask is not None, "mask is required when cond_in_channels=4 (inpainting)"
    if mask is not None and not is_inpaint:
        raise ValueError(f"mask given but cond_in_channels={cond_in_channels} (expected 4)")

    if is_inpaint and inpaint_masked_input:
        keep = (mask < 0.5).to(rgb.dtype)  # (B, 1, H, W)
        rgb = rgb * keep

    # mask channel: {0, 1} -> {-1, 1} (= (mask - 0.5) * 2). matches transforms.Normalize([0.5], [0.5])
    mask_pm1 = (mask.to(rgb.dtype) * 2.0 - 1.0) if is_inpaint else None

    if cond_input_space == "pixel":
        cond_image = torch.cat([rgb, mask_pm1], dim=1) if is_inpaint else rgb
        return cond_image, None

    assert vae is not None, "vae is required for cond_input_space='latent'"
    with torch.no_grad():
        # VAE は fp32/bf16 の自前 dtype で動かす。autocast (特に fp16) 下では conv が fp16 に
        # 落ちて NaN になりうるため明示的に無効化する。
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            cond_latent = vae.encode_pixels_to_latents(rgb.to(device=vae.device, dtype=vae.dtype))
    cond_latent = cond_latent.to(device=rgb.device, dtype=rgb.dtype)
    return cond_latent, mask_pm1


# ---------------------------------------------------------------------------
# save / load helpers
# ---------------------------------------------------------------------------
#
# 重みファイルのキー命名 (sd-scripts LoRA 互換のスタイル):
#   - 共有 conditioning encoder:    "lllite_conditioning1.{...}"
#                                   (内部の "conditioning1.{...}" を rename)
#   - 各 LLLite モジュール:         "{lllite_name}.{down|mid|cond_to_film|up}.{weight|bias}"
#                                   (lllite_name は "lllite_dit_blocks_{i}_self_attn_q_proj" 等)
#   - 各モジュールの depth embedding: "{lllite_name}.depth_embed"  shape=(cond_emb_dim,)
#                                   (内部の depth_embeds (N, D) を per-module に split)
#
# これにより、重みファイル単体から「どの DiT block のどの Linear 用か」が一意に判別できる。

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _to_saved_state_dict(lllite: "ControlNetLLLiteDiT") -> dict:
    """内部 state_dict (lllite_modules.{i}.X / conditioning1.X / depth_embeds) を
    保存用キー (lllite_name 直付け) に変換する."""
    sd = lllite.state_dict()
    names = [m.lllite_name for m in lllite.lllite_modules]
    out: dict = {}

    for k, v in sd.items():
        if k == _INTERNAL_DEPTH_KEY:
            assert v.shape[0] == len(names), (
                f"depth_embeds first dim {v.shape[0]} != n_modules {len(names)}"
            )
            for i, name in enumerate(names):
                out[f"{name}{_SAVED_DEPTH_SUFFIX}"] = v[i]
            continue
        if k.startswith(_INTERNAL_COND_PREFIX):
            out[_SAVED_COND_PREFIX + k[len(_INTERNAL_COND_PREFIX):]] = v
            continue
        if k.startswith(_INTERNAL_MODULES_PREFIX):
            rest = k[len(_INTERNAL_MODULES_PREFIX):]
            idx_str, _, suffix = rest.partition(".")
            idx = int(idx_str)
            out[f"{names[idx]}.{suffix}"] = v
            continue
        # 想定外キー (今のところ無いはず): そのまま通す
        out[k] = v

    return out


def _from_saved_state_dict(lllite: "ControlNetLLLiteDiT", weights_sd: dict) -> dict:
    """保存用キーを内部 state_dict 形式に戻す."""
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}  # idx -> (cond_emb_dim,)

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        # 未知キーはそのまま通す (load_state_dict が strict なら検出する)
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(
                f"depth_embed slices missing for module idx(es) {missing}"
            )
        out[_INTERNAL_DEPTH_KEY] = torch.stack(
            [depth_slices[i] for i in range(n_modules)], dim=0
        )

    return out


def save_lllite_model(
    file: str,
    lllite: ControlNetLLLiteDiT,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
):
    state_dict = _to_saved_state_dict(lllite)
    if dtype is not None:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu").to(dtype)
    else:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu")

    if metadata is not None and len(metadata) == 0:
        metadata = None

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    # 旧形式 (lllite_modules.{i}.X 直書きの v1 / v2 形式) は非互換なので早期 reject
    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in a legacy ControlNet-LLLite weight format "
            f"(keys starting with '{_INTERNAL_MODULES_PREFIX}'). The current code uses a "
            f"named-key format (per-module key prefix = lllite_name, e.g. "
            f"'lllite_dit_blocks_0_self_attn_q_proj.down.weight'). Re-train with the current codebase."
        )

    # pixel / latent の取り違え検出。stem のキー名が分離しているため、strict=False で
    # 黙って初期値のまま学習/推論が進むのを防ぐ (missing/unexpected は無視されてしまうため)。
    file_space = None
    if any(k.startswith(_SAVED_COND_PREFIX + "lat_conv1.") for k in weights_sd):
        file_space = "latent"
    elif any(k.startswith(_SAVED_COND_PREFIX + "conv1.") for k in weights_sd):
        file_space = "pixel"
    if file_space is not None and file_space != lllite.cond_input_space:
        raise RuntimeError(
            f"cond input space mismatch: weights at {file} were trained with "
            f"cond_input_space='{file_space}', but this LLLite was built with "
            f"cond_input_space='{lllite.cond_input_space}'. "
            f"Check --lllite_cond_input / the 'lllite.cond_input_space' metadata."
        )

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info(f"loaded LLLite weights from {file}: {info}")
    return info


# ---------------------------------------------------------------------------
# Phase A 動作確認用ダミー実行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ダミー Attention/DiT を組み立て、構築・apply_to・state_dict・forward を一通り検査する
    class _DummyAttention(nn.Module):
        def __init__(self, dim: int, ctx_dim: Optional[int]):
            super().__init__()
            self.is_selfattn = ctx_dim is None
            qd = dim
            kd = dim if ctx_dim is None else ctx_dim
            self.q_proj = nn.Linear(qd, dim, bias=False)
            self.k_proj = nn.Linear(kd, dim, bias=False)
            self.v_proj = nn.Linear(kd, dim, bias=False)
            self.output_proj = nn.Linear(dim, dim, bias=False)

        # 名前が "Attention" であることが重要 (TARGET_ATTENTION_CLASS と一致させる)

    # 実 Attention クラスを TARGET_ATTENTION_CLASS と同名にするためエイリアス
    Attention = _DummyAttention
    Attention.__name__ = "Attention"

    # GPT2FeedForward を再現するダミー MLP
    class _DummyMLP(nn.Module):
        def __init__(self, dim: int, ff_dim: int):
            super().__init__()
            self.layer1 = nn.Linear(dim, ff_dim, bias=False)
            self.layer2 = nn.Linear(ff_dim, dim, bias=False)

    DummyMLP = _DummyMLP
    DummyMLP.__name__ = "GPT2FeedForward"

    class _DummyBlock(nn.Module):
        def __init__(self, dim: int, ctx_dim: int):
            super().__init__()
            self.self_attn = Attention(dim, None)
            self.cross_attn = Attention(dim, ctx_dim)
            self.mlp = DummyMLP(dim, dim * 4)

    class _DummyDiT(nn.Module):
        def __init__(self, num_blocks: int = 4, dim: int = 64, ctx_dim: int = 128):
            super().__init__()
            self.blocks = nn.ModuleList([_DummyBlock(dim, ctx_dim) for _ in range(num_blocks)])

        def forward(self, x, t, ctx, **kwargs):
            return x

    logger.info("Phase A (v2): dummy build / apply_to / state_dict")
    NUM_BLOCKS = 4
    dit = _DummyDiT(num_blocks=NUM_BLOCKS, dim=64, ctx_dim=128)

    # parse_target_layers の単体検証
    assert parse_target_layers("self_attn_q") == ("self_attn_q_pre",)
    assert parse_target_layers("self_attn_qkv") == ("self_attn_q_pre", "self_attn_kv_pre")
    assert parse_target_layers("self_attn_qkv_cross_q") == (
        "self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre",
    )
    # canonical 順序にそろう
    assert parse_target_layers("mlp_fc1_pre,self_attn_q_pre") == ("self_attn_q_pre", "mlp_fc1_pre")
    # 重複は除去
    assert parse_target_layers("self_attn_q_pre,self_attn_q_pre") == ("self_attn_q_pre",)
    # 不正値はエラー
    try:
        parse_target_layers("bogus_atomic")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    logger.info("  parse_target_layers OK")

    cases = [
        # (spec, expected modules per block, label)
        ("self_attn_q",                                          1,  "preset self_attn_q"),
        ("self_attn_qkv",                                        3,  "preset self_attn_qkv"),
        ("self_attn_qkv_cross_q",                                4,  "preset self_attn_qkv_cross_q"),
        ("self_attn_q_pre",                                      1,  "atomic self_attn_q_pre"),
        ("mlp_fc1_pre",                                          1,  "atomic mlp_fc1_pre alone"),
        ("self_attn_q_pre,mlp_fc1_pre",                          2,  "atomic q + mlp"),
        ("self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre",         4,  "atomic qkv + mlp"),
        ("self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre,mlp_fc1_pre", 5, "all atomics"),
    ]
    for spec, per_block, label in cases:
        lllite = ControlNetLLLiteDiT(
            dit, cond_emb_dim=32, mlp_dim=64, target_layers=spec, cond_dim=64, cond_resblocks=1
        )
        expected = per_block * NUM_BLOCKS
        assert len(lllite.lllite_modules) == expected, (
            f"{label}: expected {expected} modules, got {len(lllite.lllite_modules)}"
        )
        keys = list(lllite.state_dict().keys())
        assert any(k.startswith("conditioning1.") for k in keys), keys[:5]
        assert "depth_embeds" in keys, keys
        assert any(k.startswith("lllite_modules.0.cond_to_film.") for k in keys), keys[:5]
        assert all("org_module" not in k for k in keys)
        de = lllite.state_dict()["depth_embeds"]
        assert de.shape == (expected, 32), f"{label}: depth_embeds shape mismatch: {de.shape}"
        # mlp_fc1_pre があるなら mlp_layer1 という名前のモジュールが存在するはず
        if "mlp_fc1_pre" in lllite.target_atomics:
            assert any("mlp_layer1" in m.lllite_name for m in lllite.lllite_modules), (
                f"{label}: no mlp_layer1 module found"
            )
        logger.info(f"  {label}: {len(lllite.lllite_modules)} modules OK")

    # preset と等価な atomic 表現で同じ N になる (後方互換確認)
    a = ControlNetLLLiteDiT(dit, target_layers="self_attn_qkv_cross_q")
    b = ControlNetLLLiteDiT(dit, target_layers="self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre")
    assert len(a.lllite_modules) == len(b.lllite_modules)
    assert a.target_atomics == b.target_atomics
    logger.info("  preset / atomic equivalence OK")

    # cond_resblocks=0 もサポート
    lllite_n0 = ControlNetLLLiteDiT(
        _DummyDiT(num_blocks=2, dim=64, ctx_dim=128), cond_dim=64, cond_resblocks=0
    )
    keys = list(lllite_n0.state_dict().keys())
    assert not any("resblocks" in k for k in keys), "n_resblocks=0 should not produce resblock keys"
    logger.info("  cond_resblocks=0 OK")

    # ASPP off / on の構築 + state_dict 整合
    dit_aspp = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite_no_aspp = ControlNetLLLiteDiT(dit_aspp, cond_dim=64, cond_resblocks=1, use_aspp=False)
    keys_off = list(lllite_no_aspp.state_dict().keys())
    assert not any("aspp" in k for k in keys_off), f"use_aspp=False should not produce aspp keys: {keys_off}"

    lllite_with_aspp = ControlNetLLLiteDiT(dit_aspp, cond_dim=64, cond_resblocks=1, use_aspp=True)
    keys_on = list(lllite_with_aspp.state_dict().keys())
    assert any("conditioning1.aspp.branches" in k for k in keys_on), f"use_aspp=True missing aspp keys: {keys_on[:10]}"
    assert any("conditioning1.aspp.global_conv" in k for k in keys_on)
    assert any("conditioning1.aspp.proj" in k for k in keys_on)
    # default dilations は (1,2,4,8) → branches は 4 個 + global = 5 ブランチ
    n_branches = len([k for k in keys_on if k.startswith("conditioning1.aspp.branches.") and ".0.weight" in k])
    assert n_branches == 4, f"expected 4 dilation branches, got {n_branches}"
    assert lllite_with_aspp.aspp_dilations == ASPP_DEFAULT_DILATIONS
    logger.info("  ASPP off/on state_dict OK")

    # ASPP on で zero-init forward が org_forward と一致するか (up=0 が支配)
    lllite_with_aspp.apply_to()
    wrapper_aspp = AnimaControlNetLLLiteWrapper(dit_aspp, lllite_with_aspp)
    H_a, W_a = 8, 8
    wrapper_aspp.lllite.set_cond_image(torch.randn(1, 3, H_a * 16, W_a * 16))
    mod_a = wrapper_aspp.lllite.lllite_modules[0]
    x_a = torch.randn(1, H_a * W_a, mod_a.org_module[0].in_features)
    y_a = mod_a(x_a)
    y_a_ref = mod_a.org_forward(x_a)
    assert torch.allclose(y_a, y_a_ref), "ASPP-on zero-init forward mismatch"
    logger.info("  ASPP-on zero-init forward OK")

    # 4ch (inpainting) パス: 構築 + conv1 入力チャネル + zero-init forward + save/load round-trip
    dit_4ch = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite_4ch = ControlNetLLLiteDiT(
        dit_4ch, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
        cond_dim=64, cond_resblocks=1, cond_in_channels=4, inpaint_masked_input=True,
    )
    assert lllite_4ch.cond_in_channels == 4
    assert lllite_4ch.inpaint_masked_input is True
    assert lllite_4ch.conditioning1.conv1.in_channels == 4
    # 3ch 重みは 4ch モデルにそのまま load できないが、4ch round-trip は通る
    lllite_4ch.apply_to()
    wrapper_4ch = AnimaControlNetLLLiteWrapper(dit_4ch, lllite_4ch)
    H4, W4 = 8, 8
    cond4 = torch.randn(1, 4, H4 * 16, W4 * 16)
    wrapper_4ch.lllite.set_cond_image(cond4)
    cx4 = wrapper_4ch.lllite.lllite_modules[0].cond_emb
    assert cx4 is not None and cx4.shape == (1, H4 * W4, 32), f"4ch cond_emb shape: {cx4.shape}"
    # zero-init forward
    mod4 = wrapper_4ch.lllite.lllite_modules[0]
    x4 = torch.randn(1, H4 * W4, mod4.org_module[0].in_features)
    y4 = mod4(x4)
    y4_ref = mod4.org_forward(x4)
    assert torch.allclose(y4, y4_ref), "4ch zero-init forward mismatch"
    # Wrapper の cond_image チャネル assert: 3ch を渡すと AssertionError になる
    x_lat = torch.randn(1, 16, 1, H4 * 2, W4 * 2)  # dummy latent
    try:
        # 3ch cond を渡すと拒否されるか
        wrapper_4ch(x_lat, torch.zeros(1), torch.zeros(1, 1, 1), cond_image=torch.randn(1, 3, H4 * 16, W4 * 16))
        raise AssertionError("expected channel mismatch assert")
    except AssertionError as e:
        msg = str(e)
        if "channel mismatch" not in msg and "expected 4" not in msg:
            raise
    logger.info("  4ch (inpainting) build / forward / channel assert OK")

    # ------------------------------------------------------------------
    # v2.1: latent cond input space
    # ------------------------------------------------------------------
    H_l, W_l = 8, 8                       # token grid
    LAT_H, LAT_W = H_l * 2, W_l * 2       # VAE latent 解像度 (= token * 2)
    IMG_H, IMG_W = H_l * 16, W_l * 16     # 元画像解像度 (= latent * 8)

    # 3ch 相当 (RGB のみ) の latent モード
    dit_lat = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite_lat = ControlNetLLLiteDiT(
        dit_lat, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
        cond_dim=64, cond_resblocks=1, cond_input_space="latent",
    )
    assert lllite_lat.cond_input_space == "latent"
    assert lllite_lat.conditioning1.lat_conv1.in_channels == LATENT_COND_CHANNELS
    keys_lat = list(lllite_lat.state_dict().keys())
    assert not any(k.startswith("conditioning1.conv1.") for k in keys_lat), keys_lat[:8]
    assert not any("mask_conv" in k for k in keys_lat), keys_lat[:8]
    assert any(k.startswith("conditioning1.lat_conv1.") for k in keys_lat), keys_lat[:8]
    lllite_lat.apply_to()
    wrapper_lat = AnimaControlNetLLLiteWrapper(dit_lat, lllite_lat)
    cond_lat = torch.randn(1, LATENT_COND_CHANNELS, LAT_H, LAT_W)
    wrapper_lat.lllite.set_cond_image(cond_lat)
    cx_lat = wrapper_lat.lllite.lllite_modules[0].cond_emb
    assert cx_lat is not None and cx_lat.shape == (1, H_l * W_l, 32), f"latent cond_emb shape: {cx_lat.shape}"
    mod_lat = wrapper_lat.lllite.lllite_modules[0]
    x_lat_seq = torch.randn(1, H_l * W_l, mod_lat.org_module[0].in_features)
    assert torch.allclose(mod_lat(x_lat_seq), mod_lat.org_forward(x_lat_seq)), "latent zero-init forward mismatch"
    logger.info("  latent 3ch build / set_cond_image / zero-init forward OK")

    # 4ch 相当 (RGB+mask) の latent モード: mask pyramid が生える
    dit_lat4 = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite_lat4 = ControlNetLLLiteDiT(
        dit_lat4, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
        cond_dim=64, cond_resblocks=1, cond_in_channels=4, inpaint_masked_input=True,
        cond_input_space="latent",
    )
    assert lllite_lat4.conditioning1.use_mask_branch is True
    assert lllite_lat4.conditioning1.lat_conv1.in_channels == LATENT_COND_CHANNELS * 2
    keys_lat4 = list(lllite_lat4.state_dict().keys())
    for k in ("conditioning1.mask_conv1.weight", "conditioning1.mask_conv2.weight", "conditioning1.mask_conv3.weight"):
        assert k in keys_lat4, f"missing {k}"
    assert not any("mask_norm" in k for k in keys_lat4), "mask pyramid should not have norms"
    lllite_lat4.apply_to()
    wrapper_lat4 = AnimaControlNetLLLiteWrapper(dit_lat4, lllite_lat4)
    cond_lat4 = torch.randn(1, LATENT_COND_CHANNELS, LAT_H, LAT_W)
    mask_pm1 = torch.randint(0, 2, (1, 1, IMG_H, IMG_W)).float() * 2.0 - 1.0
    wrapper_lat4.lllite.set_cond_image(cond_lat4, mask_pm1)
    cx_lat4 = wrapper_lat4.lllite.lllite_modules[0].cond_emb
    assert cx_lat4 is not None and cx_lat4.shape == (1, H_l * W_l, 32), f"latent 4ch cond_emb shape: {cx_lat4.shape}"

    # wrapper の shape assert: latent モードで pixel 解像度の cond を渡すと落ちる / mask 必須
    x_lat5d = torch.randn(1, 16, 1, LAT_H, LAT_W)
    try:
        wrapper_lat4(x_lat5d, torch.zeros(1), torch.zeros(1, 1, 1), cond_image=torch.randn(1, 3, IMG_H, IMG_W))
        raise AssertionError("expected latent cond HW/channel assert")
    except AssertionError as e:
        assert "cond latent" in str(e), str(e)
    try:
        wrapper_lat4(x_lat5d, torch.zeros(1), torch.zeros(1, 1, 1), cond_image=cond_lat4)
        raise AssertionError("expected missing cond_mask assert")
    except AssertionError as e:
        assert "cond_mask is required" in str(e), str(e)
    # 3ch latent モードに mask を渡すと拒否される
    try:
        wrapper_lat(x_lat5d, torch.zeros(1), torch.zeros(1, 1, 1), cond_image=cond_lat, cond_mask=mask_pm1)
        raise AssertionError("expected unexpected cond_mask assert")
    except AssertionError as e:
        assert "cond_mask given" in str(e), str(e)
    logger.info("  latent 4ch (mask pyramid) build / forward / wrapper asserts OK")

    # build_cond_tensors: pixel / latent 両モード
    rgb_t = torch.randn(2, 3, IMG_H, IMG_W).clamp(-1, 1)
    mask_t = (torch.rand(2, 1, IMG_H, IMG_W) > 0.5).float()

    ci_p, cm_p = build_cond_tensors(rgb_t, None, cond_input_space="pixel", cond_in_channels=3)
    assert ci_p.shape == rgb_t.shape and cm_p is None
    ci_p4, cm_p4 = build_cond_tensors(
        rgb_t, mask_t, cond_input_space="pixel", cond_in_channels=4, inpaint_masked_input=True
    )
    assert ci_p4.shape == (2, 4, IMG_H, IMG_W) and cm_p4 is None
    assert torch.allclose(ci_p4[:, 3:], mask_t * 2 - 1), "pixel mask channel must be in {-1,1}"
    assert (ci_p4[:, :3][mask_t.expand(-1, 3, -1, -1) >= 0.5] == 0).all(), "masked_input should zero RGB"

    class _FakeVae:
        """encode_pixels_to_latents だけを持つ最小のスタブ (/8 に落として 16ch にする)."""
        device = torch.device("cpu")
        dtype = torch.float32

        def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
            pooled = F.avg_pool2d(pixels, 8)  # (B, 3, H/8, W/8)
            return pooled.repeat(1, 6, 1, 1)[:, :LATENT_COND_CHANNELS]

    ci_l, cm_l = build_cond_tensors(
        rgb_t, mask_t, cond_input_space="latent", cond_in_channels=4,
        inpaint_masked_input=False, vae=_FakeVae(),
    )
    assert ci_l.shape == (2, LATENT_COND_CHANNELS, IMG_H // 8, IMG_W // 8), ci_l.shape
    assert cm_l is not None and cm_l.shape == (2, 1, IMG_H, IMG_W)
    assert cm_l.abs().eq(1.0).all(), "latent-mode mask must be in {-1,1} at pixel resolution"
    logger.info("  build_cond_tensors (pixel / latent) OK")

    # latent モードの save/load round-trip と pixel/latent 取り違え検出
    import tempfile

    tmp_lat = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False).name
    try:
        save_lllite_model(tmp_lat, lllite_lat4, dtype=torch.float32, metadata={
            "lllite.version": LLLITE_ARCH_VERSION,
            "lllite.cond_input_space": "latent",
        })
        dit_lat4_b = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
        lllite_lat4_b = ControlNetLLLiteDiT(
            dit_lat4_b, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
            cond_dim=64, cond_resblocks=1, cond_in_channels=4, inpaint_masked_input=True,
            cond_input_space="latent",
        )
        load_lllite_weights(lllite_lat4_b, tmp_lat, strict=True)
        sd_la = lllite_lat4.state_dict()
        sd_lb = lllite_lat4_b.state_dict()
        assert set(sd_la.keys()) == set(sd_lb.keys())
        for k in sd_la:
            assert torch.allclose(sd_la[k].float(), sd_lb[k].float()), f"latent round-trip mismatch at {k}"
        logger.info("  latent save / load round-trip OK")

        # latent 重みを pixel モデルに読ませると明示エラー
        dit_px = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
        lllite_px = ControlNetLLLiteDiT(
            dit_px, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
            cond_dim=64, cond_resblocks=1, cond_in_channels=4, cond_input_space="pixel",
        )
        try:
            load_lllite_weights(lllite_px, tmp_lat, strict=False)
            raise AssertionError("expected cond input space mismatch error")
        except RuntimeError as e:
            assert "cond input space mismatch" in str(e), str(e)
        logger.info("  pixel/latent weight mix-up detection OK")
    finally:
        if os.path.exists(tmp_lat):
            os.unlink(tmp_lat)

    # 非デフォルト dilations
    lllite_dil = ControlNetLLLiteDiT(
        _DummyDiT(num_blocks=2, dim=64, ctx_dim=128),
        cond_dim=64, cond_resblocks=0, use_aspp=True, aspp_dilations=(1, 3),
    )
    assert lllite_dil.aspp_dilations == (1, 3)
    keys_dil = list(lllite_dil.state_dict().keys())
    n_branches_dil = len([k for k in keys_dil if k.startswith("conditioning1.aspp.branches.") and ".0.weight" in k])
    assert n_branches_dil == 2, f"custom dilations: expected 2 branches, got {n_branches_dil}"
    logger.info("  ASPP custom dilations OK")

    # apply_to + zero-init forward
    dit2 = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite2 = ControlNetLLLiteDiT(
        dit2, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_qkv_cross_q",
        cond_dim=64, cond_resblocks=2,
    )
    lllite2.apply_to()
    wrapper = AnimaControlNetLLLiteWrapper(dit2, lllite2)

    B, H, W = 1, 8, 8
    cond_image = torch.randn(B, 3, H * 16, W * 16)
    wrapper.lllite.set_cond_image(cond_image)
    cx = wrapper.lllite.lllite_modules[0].cond_emb
    assert cx is not None and cx.shape == (B, H * W, 32), f"unexpected cond_emb shape: {cx.shape}"
    logger.info(f"  set_cond_image OK: cond_emb={tuple(cx.shape)}")

    # zero-init forward: up.weight=0 → cx=0 → org_forward(x) と一致
    mod = wrapper.lllite.lllite_modules[0]
    seq = H * W
    x_seq = torch.randn(B, seq, mod.org_module[0].in_features)
    y = mod(x_seq)
    assert y.shape == x_seq.shape
    y_ref = mod.org_forward(x_seq)
    assert torch.allclose(y, y_ref), "zero-init forward mismatch"
    logger.info("  LLLiteModuleDiT zero-init forward (3D) OK")

    # 5D 入力経路 (mlp.layer1 のような flatten されてない入力)
    # 別の DiT で mlp_fc1_pre 単独構成にして検証
    dit_mlp = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite_mlp = ControlNetLLLiteDiT(dit_mlp, target_layers="mlp_fc1_pre", cond_dim=64, cond_resblocks=0)
    lllite_mlp.apply_to()
    wrapper_mlp = AnimaControlNetLLLiteWrapper(dit_mlp, lllite_mlp)
    wrapper_mlp.lllite.set_cond_image(torch.randn(B, 3, H * 16, W * 16))
    mod_mlp = wrapper_mlp.lllite.lllite_modules[0]
    in_feat = mod_mlp.org_module[0].in_features
    out_feat = mod_mlp.org_module[0].out_features
    # T=1 固定: (B, 1, H, W, D)
    x_5d = torch.randn(B, 1, H, W, in_feat)
    y_5d = mod_mlp(x_5d)
    assert y_5d.shape == (B, 1, H, W, out_feat), f"5D output shape mismatch: {y_5d.shape}"
    y_5d_ref = mod_mlp.org_forward(x_5d)
    assert torch.allclose(y_5d, y_5d_ref), "5D zero-init forward mismatch"
    logger.info("  LLLiteModuleDiT zero-init forward (5D, mlp_fc1_pre) OK")

    # depth_embeds が non-zero でも zero-init forward は維持されるか
    # (up.weight=0 が支配的なので、depth_embeds に値を入れても出力は org_forward(x))
    with torch.no_grad():
        wrapper.lllite.depth_embeds.add_(torch.randn_like(wrapper.lllite.depth_embeds))
    y2 = mod(x_seq)
    assert torch.allclose(y2, y_ref), "up zero-init should null out non-zero depth_embeds"
    logger.info("  zero-init up dominates over depth_embeds perturbation OK")

    # save / load 互換性チェック
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp = f.name
    try:
        meta = {
            "lllite.version": LLLITE_ARCH_VERSION,
            "lllite.cond_emb_dim": "32",
            "lllite.mlp_dim": "64",
            "lllite.target_layers": "self_attn_qkv_cross_q",
            "lllite.cond_dim": "64",
            "lllite.cond_resblocks": "2",
        }
        save_lllite_model(tmp, wrapper.lllite, dtype=torch.float32, metadata=meta)

        # 保存ファイルのキー形式 (named) を検査
        from safetensors.torch import load_file as _peek_load
        saved_keys = list(_peek_load(tmp).keys())
        assert not any(k.startswith("lllite_modules.") for k in saved_keys), (
            f"saved file should not use lllite_modules.* keys: {saved_keys[:5]}"
        )
        assert any(k.startswith("lllite_conditioning1.") for k in saved_keys), saved_keys[:5]
        assert "depth_embeds" not in saved_keys, "depth_embeds should be split per-module"
        # 各 LLLite モジュールに対して named な depth_embed と down/mid/up/cond_to_film が存在する
        for m in wrapper.lllite.lllite_modules:
            assert f"{m.lllite_name}.depth_embed" in saved_keys, m.lllite_name
            assert f"{m.lllite_name}.down.weight" in saved_keys, m.lllite_name
            assert f"{m.lllite_name}.mid.weight" in saved_keys, m.lllite_name
            assert f"{m.lllite_name}.cond_to_film.weight" in saved_keys, m.lllite_name
            assert f"{m.lllite_name}.up.weight" in saved_keys, m.lllite_name
        logger.info("  saved key format (named) OK")

        dit3 = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
        lllite3 = ControlNetLLLiteDiT(
            dit3, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_qkv_cross_q",
            cond_dim=64, cond_resblocks=2,
        )
        load_lllite_weights(lllite3, tmp, strict=True)
        # round-trip 後、内部 state_dict が完全一致
        sd_orig = wrapper.lllite.state_dict()
        sd_loaded = lllite3.state_dict()
        assert set(sd_orig.keys()) == set(sd_loaded.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k].float(), sd_loaded[k].float()), f"mismatch at {k}"
        logger.info("  save / load round-trip OK")

        # 4ch round-trip
        tmp4 = tmp + ".4ch.safetensors"
        try:
            meta4 = {
                "lllite.version": LLLITE_ARCH_VERSION,
                "lllite.cond_emb_dim": "32",
                "lllite.mlp_dim": "64",
                "lllite.target_layers": "self_attn_q",
                "lllite.cond_dim": "64",
                "lllite.cond_resblocks": "1",
                "lllite.cond_in_channels": "4",
                "lllite.inpaint_masked_input": "true",
            }
            save_lllite_model(tmp4, lllite_4ch, dtype=torch.float32, metadata=meta4)
            dit_4ch_b = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
            lllite_4ch_b = ControlNetLLLiteDiT(
                dit_4ch_b, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
                cond_dim=64, cond_resblocks=1, cond_in_channels=4, inpaint_masked_input=True,
            )
            load_lllite_weights(lllite_4ch_b, tmp4, strict=True)
            sd_a = lllite_4ch.state_dict()
            sd_b = lllite_4ch_b.state_dict()
            assert set(sd_a.keys()) == set(sd_b.keys())
            for k in sd_a:
                assert torch.allclose(sd_a[k].float(), sd_b[k].float()), f"4ch round-trip mismatch at {k}"
            logger.info("  4ch save / load round-trip OK")
        finally:
            if os.path.exists(tmp4):
                os.unlink(tmp4)

        # 旧形式 (lllite_modules.* キー) は reject される
        legacy_sd = {"lllite_modules.0.up.weight": torch.zeros(1)}
        from safetensors.torch import save_file as _save_legacy
        legacy_tmp = tmp + ".legacy.safetensors"
        _save_legacy(legacy_sd, legacy_tmp)
        try:
            try:
                load_lllite_weights(lllite3, legacy_tmp, strict=False)
                raise AssertionError("legacy format should be rejected")
            except RuntimeError as e:
                assert "legacy" in str(e).lower()
            logger.info("  legacy format reject OK")
        finally:
            os.unlink(legacy_tmp)
    finally:
        os.unlink(tmp)

    logger.info("Phase A (v2) dummy check PASSED")
