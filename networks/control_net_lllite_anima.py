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
# v3 (semantic trunk) のアーキテクチャ世代
LLLITE_ARCH_VERSION_SEMANTIC = "3"

# cond 画像の入力空間 (v2.1 で追加。メタデータ欠落時は "pixel" = 旧重み互換)
COND_INPUT_SPACES: Tuple[str, ...] = ("pixel", "latent")

# conditioning trunk の種類 (v3 で追加。メタデータ欠落時は "stem" = 旧重み互換)
#   stem    : cond 入力をスクラッチ conv stem に通す (v2 / v2.1)
#   semantic: cond latent を凍結 DiT に通した中間 hidden states を条件源にする (v3)
TRUNK_TYPES: Tuple[str, ...] = ("stem", "semantic")

# latent モードで conditioning1 が受け取る VAE latent のチャネル数 (Qwen-Image VAE z_dim)
LATENT_COND_CHANNELS = 16

# v3 gate の bias 初期値。ゲートは「開いた状態」で初期化する (閉じて初期化すると値パスへの
# 勾配が 0 倍されて学習が立ち上がらない)。σ(2.0) ≈ 0.88
GATE_INIT_BIAS = 2.0


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


def parse_ref_blocks(spec) -> Optional[Tuple[int, ...]]:
    """ref_block 指定を canonical な tuple に解決する (v3 semantic trunk).

    受理する形式:
      - None (デフォルト解決を後段 = ControlNetLLLiteDiT に委ねる)
      - int 単体 (v3 single)
      - int の sequence / カンマ区切り文字列 "2,13" (v3 dual/multi concat)

    index -1 は特別値で、x_embedder 出力 (patchify 直後・DiT ブロック通過前) =
    cond latent の線形埋め込みそのものを条件源にする (「block -1 の出力 = block 0 の入力」)。
    純外観のコピー信号で、テキスト context / ref_timestep に依存しない。

    返り値は昇順・重複なしの tuple。concat の並び順はこの昇順で固定される
    (保存重み proj_in の列レイアウトと一致させるため)。-1 は昇順で常に先頭になる。
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        blocks = [spec]
    elif isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("ref_block spec is empty")
        blocks = [int(p) for p in parts]
    else:
        blocks = [int(b) for b in spec]
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"duplicate ref_block index in {blocks}")
    return tuple(sorted(blocks))


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


class _ConditioningSemanticTrunk(nn.Module):
    """v3 semantic trunk: 凍結 DiT の中間 hidden states を条件源にする.

      in  h_ref (B, T=1, H, W, D_model)          # encode_reference_hidden_states の出力 (token 解像度)
          または (B, K, T=1, H, W, D_model)       # dual/multi: K 個の ref block を concat
      -> LayerNorm(D_model)  [per-block]         # 層ごとのスケール差を吸収
      -> concat (K*D_model) -> Linear -> cond_dim # 共有 down 射影 (「選択と整列」なので低ランクで足りる)
      -> (B, cond_dim, H, W)
      -> ResBlock x N / (ASPP)                   # 中域文脈の拡張 (stem trunk と同じ部品)
      -> Conv 1x1 -> cond_emb_dim
      -> flatten (B, S, cond_emb_dim) -> LayerNorm

    num_ref_blocks == 1 のとき ln_in は素の LayerNorm、proj_in は D_model -> cond_dim で
    v3 single の保存キー・形状と完全互換。K > 1 のとき ln_in は per-block の ModuleList、
    proj_in は K*D_model -> cond_dim になる (single 重みとは非互換、ロード時に検出される)。

    t_proj は本体の timestep embedding (t_embedding_norm 出力 (B, T, D_model)) を
    cond_emb_dim へ落とす zero-init 射影。cond_local に加算され、既存の FiLM / mid / gate が
    cond_local 経由で t 条件を受け取る (t-FiLM)。
    """

    def __init__(
        self,
        model_dim: int,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        num_ref_blocks: int = 1,
    ):
        super().__init__()
        assert num_ref_blocks >= 1, f"num_ref_blocks must be >= 1, got {num_ref_blocks}"
        self.num_ref_blocks = num_ref_blocks

        if num_ref_blocks == 1:
            self.ln_in = nn.LayerNorm(model_dim)
            self.proj_in = nn.Linear(model_dim, cond_dim)
        else:
            # per-block LayerNorm: LN 自体はトークン毎の正規化なので共有でも数値は同じだが、
            # affine をブロック毎に持たせて深さの違う特徴のスケール/シフトを独立に学ばせる
            self.ln_in = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_ref_blocks)])
            self.proj_in = nn.Linear(model_dim * num_ref_blocks, cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

        # t は zero-init で「t 非依存」から学習を開始する
        self.t_proj = nn.Linear(model_dim, cond_emb_dim)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

    def forward(self, h_ref: torch.Tensor) -> torch.Tensor:
        # (B, T, H, W, D) [K=1] または (B, K, T, H, W, D) を受理し、K 次元へ正規化する
        if h_ref.dim() == 5:
            h_ref = h_ref.unsqueeze(1)
        assert h_ref.dim() == 6, (
            f"semantic trunk expects (B, T, H, W, D) or (B, K, T, H, W, D), got {tuple(h_ref.shape)}"
        )
        b, k, t, hh, ww, _ = h_ref.shape
        assert k == self.num_ref_blocks, (
            f"semantic trunk was built for {self.num_ref_blocks} ref block(s) but got {k} "
            f"(check ref_block vs the trained weights)"
        )
        assert t == 1, f"semantic trunk supports T=1 only, got T={t}"
        if self.num_ref_blocks == 1:
            h = self.proj_in(self.ln_in(h_ref[:, 0]))  # (B, 1, H, W, cond_dim)
        else:
            normed = [ln(h_ref[:, i]) for i, ln in enumerate(self.ln_in)]
            h = self.proj_in(torch.cat(normed, dim=-1))  # (B, 1, H, W, cond_dim)
        h = h.squeeze(1).permute(0, 3, 1, 2).contiguous()  # (B, cond_dim, H, W)
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)  # (B, cond_emb_dim, H, W)
        c = h.shape[1]
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()  # (B, S, cond_emb_dim)
        return self.out_norm(h)


class LLLiteModuleDiT(nn.Module):
    """単一の Attention Linear (q_proj/k_proj/v_proj) に対し LLLite の補正 x + cx を注入する.

    v2: concat-then-mid をベースに FiLM (γ, β) を mid 出力に適用、SiLU 化、depth embedding 対応.
    v3 (use_gate=True): さらに per-token スカラーゲート g = σ(gate([cond_local, h])) を持ち、
    出力を Δx = g ⊙ up(m) にする (値パス=何を注入するか / ゲート=どこで注入するか の分解)。
    """

    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        use_gate: bool = False,
    ):
        super().__init__()
        self.lllite_name = name
        # list 包みで nn.Module 登録を回避し、state_dict に元 Linear の重みが入らないようにする
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier
        self.use_gate = use_gate

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

        if use_gate:
            # per-token スカラーゲート。weight は zero-init (空間一様スタート)、bias は
            # 開いた状態 (GATE_INIT_BIAS)。値パス up が zero-init なので恒等スタートは維持され、
            # 開いたゲート越しに値パスへ勾配が流れ、ゲートは後から「閉じるべき場所」を学ぶ。
            self.gate = nn.Linear(cond_emb_dim + mlp_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, GATE_INIT_BIAS)

        # 親 ControlNetLLLiteDiT が set_cond_image で注入する。
        # cond_emb は全モジュールで共有される cx (B, S, cond_emb_dim)、
        # depth_emb はこのモジュール用の depth embedding (cond_emb_dim,)。
        # cx を共有参照することで N コピーを避け、加算は forward 内で行う。
        self.cond_emb: Optional[torch.Tensor] = None
        self.depth_emb: Optional[torch.Tensor] = None

        # v3 (semantic trunk): dit.t_embedding_norm の forward hook が毎 forward 更新する
        # 共有 t 埋め込み (B, T, cond_emb_dim)。stem trunk では常に None。
        self.t_local: Optional[torch.Tensor] = None

        # gate 可視化用 (推論時のみ有効化する想定。capture_gate=True のとき forward 毎に
        # last_gate へ detach 済みゲートマップ (B, S, 1) を保存する)
        self.capture_gate: bool = False
        self.last_gate: Optional[torch.Tensor] = None

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

        # v3 (semantic trunk): per-step の timestep 埋め込みを cond 空間で加算する (t-FiLM)。
        # t_local は dit.t_embedding_norm の forward hook が blocks 実行前に毎 forward 更新する。
        t_local = self.t_local
        if self.use_gate:
            assert t_local is not None, (
                f"t_local is not set ({self.lllite_name}); the semantic trunk requires the "
                "timestep hook on dit.t_embedding_norm (registered by ControlNetLLLiteDiT) "
                "to fire before the blocks run"
            )
        if t_local is not None and t_local.shape[0] == cond_local.shape[0]:
            cond_local = cond_local + t_local  # (B, T, D) broadcast over S
            t_local = None

        # CFG 推論用 (学習時は通らない想定)
        if x.shape[0] // 2 == cond_local.shape[0]:
            cond_local = cond_local.repeat(2, 1, 1)

        # CFG をバッチ化した推論 (x が 2B, t_local が 2B) では repeat 後に加算する
        if t_local is not None:
            assert t_local.shape[0] == cond_local.shape[0], (
                f"LLLite t_local batch mismatch ({self.lllite_name}): "
                f"t_local={t_local.shape[0]} vs cond={cond_local.shape[0]}"
            )
            cond_local = cond_local + t_local

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

        out = self.up(m)

        if self.use_gate:
            # per-token スカラーゲート: cond の意味特徴と生成側の現在特徴 h の両方を見て
            # 「この位置に条件を注入するか」を内容依存に決める
            g = torch.sigmoid(self.gate(torch.cat([cond_local, h], dim=-1)))  # (B, S, 1)
            if self.capture_gate:
                self.last_gate = g.detach()
            out = out * g

        out = out * self.multiplier
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
        trunk: str = "stem",
        ref_block=None,  # int / "2,13" 形式の str / int sequence (parse_ref_blocks 参照)
        ref_timestep: float = 0.0,
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)
        assert cond_input_space in COND_INPUT_SPACES, (
            f"cond_input_space must be one of {list(COND_INPUT_SPACES)}, got {cond_input_space!r}"
        )
        assert trunk in TRUNK_TYPES, f"trunk must be one of {list(TRUNK_TYPES)}, got {trunk!r}"

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
        # "stem": conditioning1 = conv stem (v2/v2.1) / "semantic": 凍結 DiT hidden states (v3)
        self.trunk = trunk

        if trunk == "semantic":
            # cond latent を凍結 DiT に通すため latent 入力が前提。mask (inpaint) は MVP 未対応
            assert cond_input_space == "latent", (
                "trunk='semantic' requires cond_input_space='latent' "
                "(the cond latent is fed through the frozen DiT)"
            )
            assert cond_in_channels == 3, (
                f"trunk='semantic' does not support inpainting (cond_in_channels=4) yet, "
                f"got cond_in_channels={cond_in_channels}"
            )

            model_dim = getattr(dit, "model_channels", None)
            if model_dim is None:
                model_dim = getattr(dit.blocks[0], "x_dim", None)
            assert model_dim is not None, (
                "cannot infer the DiT hidden dim (model_channels / blocks[0].x_dim) for the semantic trunk"
            )
            num_dit_blocks = len(dit.blocks)
            ref_blocks = parse_ref_blocks(ref_block)
            if ref_blocks is None:
                ref_blocks = (num_dit_blocks // 2,)
            for rb in ref_blocks:
                assert -1 <= rb < num_dit_blocks, (
                    f"ref_block {rb} out of range (-1 = x_embedder output, 0..{num_dit_blocks - 1} = block output)"
                )
            self.model_dim = int(model_dim)
            self.ref_blocks = ref_blocks
            self.ref_timestep = float(ref_timestep)

            # semantic trunk: hidden states (B, [K,] T, H, W, D_model) -> (B, S, cond_emb_dim)
            self.conditioning1 = _ConditioningSemanticTrunk(
                self.model_dim, cond_dim, cond_emb_dim, cond_resblocks,
                use_aspp=use_aspp, aspp_dilations=aspp_dilations,
                num_ref_blocks=len(ref_blocks),
            )
        else:
            self.model_dim = None
            self.ref_blocks = None
            self.ref_timestep = 0.0

            # pixel : cond image  (B, cond_in_channels, H*16, W*16) -> (B, S, cond_emb_dim)
            # latent: cond latent (B, 16, H*2, W*2) (+ mask (B,1,H*16,W*16)) -> (B, S, cond_emb_dim)
            self.conditioning1 = _Conditioning1(
                cond_dim, cond_emb_dim, cond_resblocks,
                use_aspp=use_aspp, aspp_dilations=aspp_dilations,
                cond_in_channels=cond_in_channels,
                input_space=cond_input_space,
            )

        modules = self._create_modules(
            dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier, use_gate=trunk == "semantic"
        )
        self.lllite_modules = nn.ModuleList(modules)

        # depth embedding: 各モジュール用の zero-init bias (N, cond_emb_dim)
        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i

        # gate 可視化用: 直近の set_cond の token grid (H, W)
        self.last_cond_hw: Optional[Tuple[int, int]] = None

        # v3: 本体の timestep embedding を per-step で受け取る hook。t_embedding_norm は
        # blocks の実行前に呼ばれるため、学習・推論どちらのループでも無改造で t が届く。
        self._t_hook_handle = None
        if trunk == "semantic":
            t_norm = getattr(dit, "t_embedding_norm", None)
            if t_norm is not None:
                def _t_hook(_module, _inputs, output, _self=self):
                    _self._update_t_local(output)
                    return None

                self._t_hook_handle = t_norm.register_forward_hook(_t_hook)
            else:
                logger.warning(
                    "dit has no t_embedding_norm; semantic trunk t-FiLM hook is not registered "
                    "(call _update_t_local manually before each forward)"
                )

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        inpaint_info = (
            f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels == 4 else ""
        )
        trunk_info = (
            f"trunk=semantic(ref_blocks={list(self.ref_blocks)}, ref_timestep={self.ref_timestep}, "
            f"model_dim={self.model_dim})"
            if trunk == "semantic"
            else "trunk=stem"
        )
        version = LLLITE_ARCH_VERSION_SEMANTIC if trunk == "semantic" else LLLITE_ARCH_VERSION
        logger.info(
            f"ControlNet-LLLite (Anima v{version}): created {n} modules for "
            f"target={target_layers!r} (atomics={list(atomics)}), "
            f"{trunk_info}, cond_input={cond_input_space}, "
            f"cond_in_channels={cond_in_channels}, cond_dim={cond_dim}, cond_resblocks={cond_resblocks}, {aspp_info}, "
            f"cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}{inpaint_info}"
        )

    @property
    def target_atomics_str(self) -> str:
        """canonical atomic specifier をカンマ区切り文字列で返す (メタデータ保存用)."""
        return ",".join(self.target_atomics)

    @property
    def ref_blocks_str(self) -> str:
        """ref_blocks をカンマ区切り文字列で返す (メタデータ保存用。semantic trunk のみ)."""
        assert self.ref_blocks is not None, "ref_blocks_str is only for trunk='semantic'"
        return ",".join(str(b) for b in self.ref_blocks)

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
        use_gate: bool = False,
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
                        LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier, use_gate)
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                # GPT2FeedForward.layer1 = fc1 (d_model -> d_ff)
                child = getattr(module, "layer1", None)
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier, use_gate)
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor], cond_mask: Optional[torch.Tensor] = None):
        """cond_image を conditioning1 に通し、全 LLLite モジュールに cond_emb を配る。None で解除。

        pixel モード : cond_image = (B, cond_in_channels, H*16, W*16)、cond_mask は不可
        latent モード: cond_image = (B, 16, H*2, W*2) の正規化済み VAE latent、
                       inpaint (cond_in_channels=4) のとき cond_mask = (B, 1, H*16, W*16) in [-1, 1]

        semantic trunk (v3) では代わりに set_cond_hidden_states を使う (None での解除は共通)。
        """
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
                m.depth_emb = None
                m.t_local = None
            return
        assert self.trunk == "stem", (
            "set_cond_image with a tensor is only for the stem trunk; "
            "use set_cond_hidden_states (with encode_reference_hidden_states) for trunk='semantic'"
        )
        cx = self.conditioning1(cond_image, cond_mask)  # (B, S, cond_emb_dim)
        self._distribute_cond_emb(cx)

    def set_cond_hidden_states(self, hidden_states: torch.Tensor):
        """v3 (semantic trunk): encode_reference_hidden_states の出力を semantic trunk に通し、
        全 LLLite モジュールに cond_emb を配る。

        single (K=1): (B, T=1, H, W, D_model) / dual・multi: (B, K, T=1, H, W, D_model)。
        K の検証は trunk の forward が行う。"""
        assert self.trunk == "semantic", "set_cond_hidden_states is only for trunk='semantic'"
        cx = self.conditioning1(hidden_states)  # (B, S, cond_emb_dim)
        self.last_cond_hw = (hidden_states.shape[-3], hidden_states.shape[-2])
        self._distribute_cond_emb(cx)

    def _distribute_cond_emb(self, cx: torch.Tensor):
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

    def _update_t_local(self, t_emb: torch.Tensor):
        """v3: dit.t_embedding_norm の出力 (B, T, D_model) を t_proj で cond 空間に落とし、
        全モジュールへ配る。dit.t_embedding_norm の forward hook から毎 forward 呼ばれる。"""
        t_proj = self.conditioning1.t_proj
        # t_embedder は凍結だが、detach で graph を確実に切る (t_proj 側には勾配が流れる)
        t_local = t_proj(t_emb.detach().to(t_proj.weight.dtype))  # (B, T, cond_emb_dim)
        for m in self.lllite_modules:
            m.t_local = t_local

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()


def encode_reference_hidden_states(
    dit: nn.Module,
    cond_latent: torch.Tensor,
    ref_block,
    ref_timestep: float = 0.0,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """v3 (semantic trunk): 凍結 Anima DiT に cond latent を通し、blocks[ref_block] 出力の
    hidden states を返す。max(ref_block) 以降のブロックは実行しない。

    dual/multi (sequence 指定) では 1 回の参照フォワードで各 index の出力を収集する。
    実行コストは max(ref_block) で決まるため、浅いブロックの追加コストはゼロ。

    index -1 は x_embedder 出力 (patchify 直後・ブロック通過前) を返す特別値。
    -1 のみの指定ではブロックを一切実行しない (t 埋め込み・context も不要)。

    cross-attn の context にはゼロテンソルを渡す。Anima は padding トークンをゼロ潰しして
    そのまま attend させる規約で、かつ k_proj/v_proj は bias なしなので、ゼロ context では
    cross-attn の出力が厳密に 0 になる (= テキスト完全非依存の「画像のみの特徴」)。
    テキストに依存しないため、将来のディスクキャッシュ対象にできる。

    呼び出し側の責務:
      - 事前に LLLite の cond を clear しておく (モジュールは cond_emb None で素通りする)
      - 勾配は不要なので torch.no_grad() 下で呼ぶ

    Args:
        dit: 凍結 Anima モデル
        cond_latent: (B, 16, h, w) or (B, 16, 1, h, w) の正規化済み VAE latent
        ref_block: hidden states を取り出すブロック index。int 単体、または
                   sequence / "2,13" 形式の str (dual/multi concat、昇順に正規化される)
        ref_timestep: 参照フォワードの timestep。学習ループと同じ [0, 1] スケール (0 = clean)
        padding_mask: (B, 1, h, w)。None ならゼロ

    Returns:
        int 指定:              (B, 1, H, W, D_model) hidden states (token 解像度 = latent の 1/2)
        sequence / str 指定:   (B, K, 1, H, W, D_model)、K 次元は ref_block の昇順
    """
    from library import attention as attention_lib

    single = isinstance(ref_block, int)
    ref_blocks = parse_ref_blocks(ref_block)
    assert ref_blocks is not None and len(ref_blocks) >= 1, f"invalid ref_block: {ref_block!r}"

    if cond_latent.dim() == 4:
        cond_latent = cond_latent.unsqueeze(2)  # (B, 16, 1, h, w)
    assert cond_latent.dim() == 5 and cond_latent.shape[2] == 1, (
        f"cond_latent must be (B, C, 1, h, w), got {tuple(cond_latent.shape)}"
    )
    bsz = cond_latent.shape[0]
    h_lat, w_lat = cond_latent.shape[-2], cond_latent.shape[-1]
    if padding_mask is None:
        padding_mask = torch.zeros(bsz, 1, h_lat, w_lat, device=cond_latent.device, dtype=cond_latent.dtype)

    x, rope_emb_L_1_1_D, extra_pos_emb = dit.prepare_embedded_sequence(
        cond_latent, fps=None, padding_mask=padding_mask
    )

    assert -1 <= ref_blocks[0] and ref_blocks[-1] < len(dit.blocks), (
        f"ref_block {list(ref_blocks)} out of range (num_blocks={len(dit.blocks)}, -1 = x_embedder output)"
    )
    ref_set = set(ref_blocks)
    max_block = ref_blocks[-1]
    collected: List[torch.Tensor] = []
    if -1 in ref_set:
        # "block -1" = x_embedder 出力 (ブロック通過前)。昇順収集なので常に先頭
        collected.append(x)

    if max_block >= 0:
        t = torch.full((bsz, 1), float(ref_timestep), device=cond_latent.device, dtype=cond_latent.dtype)
        t_embedding_B_T_D, adaln_lora_B_T_3D = dit.t_embedder(t)
        t_embedding_B_T_D = dit.t_embedding_norm(t_embedding_B_T_D)

        ctx_dim = dit.blocks[0].cross_attn.context_dim
        context = torch.zeros(bsz, 1, ctx_dim, device=cond_latent.device, dtype=cond_latent.dtype)

        attn_params = attention_lib.AttentionParams.create_attention_params(dit.attn_mode, dit.split_attn)
        use_fp32 = x.dtype == torch.float16

        for block_idx, block in enumerate(dit.blocks):
            x = block(
                x,
                t_embedding_B_T_D,
                context,
                attn_params,
                use_fp32,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )
            if block_idx in ref_set:
                collected.append(x)
            if block_idx == max_block:
                break
    assert len(collected) == len(ref_blocks)
    if single:
        return collected[0]  # (B, T, H, W, D)
    return torch.stack(collected, dim=1)  # (B, K, T, H, W, D)


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
            if self.lllite.trunk == "semantic":
                # v3: cond latent を凍結 DiT に通し、hidden states を semantic trunk へ渡す
                assert cond_mask is None, (
                    "cond_mask is not supported with trunk='semantic' (v3 MVP)"
                )
                assert cond_image.shape[-2] == latent_h and cond_image.shape[-1] == latent_w, (
                    f"cond latent HW mismatch: expected {latent_h}x{latent_w} (same as x), "
                    f"got {cond_image.shape[-2]}x{cond_image.shape[-1]}"
                )
                assert cond_image.shape[1] == LATENT_COND_CHANNELS, (
                    f"cond latent channel mismatch: expected {LATENT_COND_CHANNELS}, "
                    f"got {cond_image.shape[1]}"
                )
                # 参照フォワード中は LLLite モジュールを素通りさせる
                self.lllite.clear_cond_image()
                with torch.no_grad():
                    h_ref = encode_reference_hidden_states(
                        self.dit,
                        cond_image,
                        self.lllite.ref_blocks,
                        self.lllite.ref_timestep,
                        padding_mask=kwargs.get("padding_mask"),
                    )
                self.lllite.set_cond_hidden_states(h_ref)
                return self.dit(x, timesteps, context, **kwargs)
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

    # trunk (stem/semantic) の取り違え検出。conditioning のキー名が分離しているため、
    # strict=False で黙って初期値のまま学習/推論が進むのを防ぐ。
    file_trunk = None
    if any(k.startswith(_SAVED_COND_PREFIX + "proj_in.") for k in weights_sd):
        file_trunk = "semantic"
    elif any(
        k.startswith(_SAVED_COND_PREFIX + "lat_conv1.") or k.startswith(_SAVED_COND_PREFIX + "conv1.")
        for k in weights_sd
    ):
        file_trunk = "stem"
    if file_trunk is not None and file_trunk != lllite.trunk:
        raise RuntimeError(
            f"trunk mismatch: weights at {file} were trained with trunk='{file_trunk}', "
            f"but this LLLite was built with trunk='{lllite.trunk}'. "
            f"Check --lllite_trunk / the 'lllite.trunk' metadata."
        )

    # semantic trunk の single / dual (ref block 数) 取り違え検出。
    # proj_in の形状不一致でもロードは失敗するが、原因が ref_block 指定だと分かる形で早期に落とす。
    if file_trunk == "semantic" and lllite.trunk == "semantic":
        if any(k == _SAVED_COND_PREFIX + "ln_in.weight" for k in weights_sd):
            file_k = 1
        else:
            ln_prefix = _SAVED_COND_PREFIX + "ln_in."
            file_k = len(
                {k[len(ln_prefix):].split(".")[0] for k in weights_sd if k.startswith(ln_prefix)}
            )
        model_k = len(lllite.ref_blocks)
        if file_k != model_k:
            raise RuntimeError(
                f"ref block count mismatch: weights at {file} were trained with {file_k} ref block(s), "
                f"but this LLLite was built with ref_blocks={list(lllite.ref_blocks)} ({model_k}). "
                f"Check --lllite_ref_block / the 'lllite.ref_block' metadata."
            )

    # pixel / latent の取り違え検出 (stem trunk のみ。semantic は latent 固定)。
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

    # ------------------------------------------------------------------
    # v3: semantic trunk (dummy 版。実 Anima での end-to-end は
    # tools/dev/manual_test_anima_lllite_dryrun.py の check_semantic_trunk で検査する)
    # ------------------------------------------------------------------
    MODEL_DIM = 64

    class _DummyDiTV3(_DummyDiT):
        def __init__(self, num_blocks: int = 4, dim: int = MODEL_DIM, ctx_dim: int = 128):
            super().__init__(num_blocks=num_blocks, dim=dim, ctx_dim=ctx_dim)
            self.model_channels = dim
            # 実 Anima では t_embedding_norm の forward hook が t_local を配る。
            # dummy では _update_t_local を手動で呼ぶ (hook 登録の warning 経路も同時に検査)。

    dit_v3 = _DummyDiTV3(num_blocks=4)
    lllite_v3 = ControlNetLLLiteDiT(
        dit_v3, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_qkv_cross_q",
        cond_dim=64, cond_resblocks=1,
        cond_input_space="latent", trunk="semantic", ref_block=2, ref_timestep=0.0,
    )
    assert lllite_v3.trunk == "semantic"
    assert lllite_v3.ref_blocks == (2,)
    assert lllite_v3.ref_blocks_str == "2"
    assert lllite_v3.model_dim == MODEL_DIM
    keys_v3 = list(lllite_v3.state_dict().keys())
    assert any(k.startswith("conditioning1.proj_in.") for k in keys_v3), keys_v3[:8]
    assert any(k.startswith("conditioning1.t_proj.") for k in keys_v3), keys_v3[:8]
    assert any(k.endswith(".gate.weight") for k in keys_v3), keys_v3[:8]
    assert not any("lat_conv" in k or "conditioning1.conv1." in k for k in keys_v3), keys_v3[:8]
    # gate 初期化: weight=0, bias=GATE_INIT_BIAS (開いた状態)
    g0 = lllite_v3.lllite_modules[0].gate
    assert g0.weight.abs().sum().item() == 0.0
    assert torch.allclose(g0.bias, torch.full_like(g0.bias, GATE_INIT_BIAS))
    logger.info("  v3 semantic trunk build / keys / gate init OK")

    # ref_block デフォルト = num_blocks // 2
    lllite_v3_d = ControlNetLLLiteDiT(
        _DummyDiTV3(num_blocks=4), cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
        cond_input_space="latent", trunk="semantic",
    )
    assert lllite_v3_d.ref_blocks == (2,), lllite_v3_d.ref_blocks
    logger.info("  v3 ref_block default (num_blocks // 2) OK")

    # parse_ref_blocks の単体検証
    assert parse_ref_blocks(None) is None
    assert parse_ref_blocks(13) == (13,)
    assert parse_ref_blocks("13") == (13,)
    assert parse_ref_blocks("2,13") == (2, 13)
    assert parse_ref_blocks("13, 2") == (2, 13)  # 昇順に正規化
    assert parse_ref_blocks([13, 2]) == (2, 13)
    for bad in ("", "2,2", [3, 3]):
        try:
            parse_ref_blocks(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass
    logger.info("  parse_ref_blocks OK")

    # semantic は latent 入力必須 / inpaint (4ch) 非対応
    try:
        ControlNetLLLiteDiT(_DummyDiTV3(), trunk="semantic", cond_input_space="pixel")
        raise AssertionError("expected latent-required assert")
    except AssertionError as e:
        assert "requires cond_input_space='latent'" in str(e), str(e)
    try:
        ControlNetLLLiteDiT(_DummyDiTV3(), trunk="semantic", cond_input_space="latent", cond_in_channels=4)
        raise AssertionError("expected inpaint-unsupported assert")
    except AssertionError as e:
        assert "does not support inpainting" in str(e), str(e)
    logger.info("  v3 constraint asserts OK")

    # set_cond_hidden_states + _update_t_local + zero-init forward
    lllite_v3.apply_to()
    H_v3, W_v3 = 8, 8
    h_ref = torch.randn(1, 1, H_v3, W_v3, MODEL_DIM)
    t_emb = torch.randn(1, 1, MODEL_DIM)
    lllite_v3.set_cond_hidden_states(h_ref)
    lllite_v3._update_t_local(t_emb)
    assert lllite_v3.last_cond_hw == (H_v3, W_v3)
    mod_v3 = lllite_v3.lllite_modules[0]
    assert mod_v3.cond_emb is not None and mod_v3.cond_emb.shape == (1, H_v3 * W_v3, 32)
    assert mod_v3.t_local is not None and mod_v3.t_local.shape == (1, 1, 32)
    x_v3 = torch.randn(1, H_v3 * W_v3, mod_v3.org_module[0].in_features)
    y_v3 = mod_v3(x_v3)
    assert torch.allclose(y_v3, mod_v3.org_forward(x_v3)), "v3 zero-init forward mismatch"
    # gate capture
    mod_v3.capture_gate = True
    mod_v3(x_v3)
    assert mod_v3.last_gate is not None and mod_v3.last_gate.shape == (1, H_v3 * W_v3, 1)
    # zero-init では gate は空間一様に σ(GATE_INIT_BIAS)
    expected_g = torch.sigmoid(torch.tensor(GATE_INIT_BIAS))
    assert torch.allclose(mod_v3.last_gate, expected_g.expand_as(mod_v3.last_gate), atol=1e-6)
    mod_v3.capture_gate = False
    # clear で t_local も落ちる
    lllite_v3.clear_cond_image()
    assert mod_v3.cond_emb is None and mod_v3.t_local is None
    logger.info("  v3 set_cond_hidden_states / t_local / zero-init forward / gate capture OK")

    # CFG 系バッチ整合: cond B=1, x 2B (t_local B=1 は repeat 前に加算される)
    lllite_v3.set_cond_hidden_states(h_ref)
    lllite_v3._update_t_local(t_emb)
    x_cfg = torch.randn(2, H_v3 * W_v3, mod_v3.org_module[0].in_features)
    y_cfg = mod_v3(x_cfg)
    assert torch.allclose(y_cfg, mod_v3.org_forward(x_cfg)), "v3 CFG-batch zero-init mismatch"
    lllite_v3.clear_cond_image()
    logger.info("  v3 CFG batch (cond B=1, x 2B) OK")

    # save / load round-trip + trunk 取り違え検出
    tmp_v3 = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False).name
    try:
        save_lllite_model(tmp_v3, lllite_v3, dtype=torch.float32, metadata={
            "lllite.version": LLLITE_ARCH_VERSION_SEMANTIC,
            "lllite.trunk": "semantic",
            "lllite.cond_input_space": "latent",
        })
        lllite_v3_b = ControlNetLLLiteDiT(
            _DummyDiTV3(num_blocks=4), cond_emb_dim=32, mlp_dim=64,
            target_layers="self_attn_qkv_cross_q", cond_dim=64, cond_resblocks=1,
            cond_input_space="latent", trunk="semantic", ref_block=2,
        )
        load_lllite_weights(lllite_v3_b, tmp_v3, strict=True)
        sd_v3a = lllite_v3.state_dict()
        sd_v3b = lllite_v3_b.state_dict()
        assert set(sd_v3a.keys()) == set(sd_v3b.keys())
        for k in sd_v3a:
            assert torch.allclose(sd_v3a[k].float(), sd_v3b[k].float()), f"v3 round-trip mismatch at {k}"
        logger.info("  v3 save / load round-trip OK")

        # semantic 重みを stem (latent) モデルに読ませると明示エラー
        lllite_stem = ControlNetLLLiteDiT(
            _DummyDiTV3(num_blocks=4), cond_emb_dim=32, mlp_dim=64,
            target_layers="self_attn_qkv_cross_q", cond_dim=64, cond_resblocks=1,
            cond_input_space="latent", trunk="stem",
        )
        try:
            load_lllite_weights(lllite_stem, tmp_v3, strict=False)
            raise AssertionError("expected trunk mismatch error")
        except RuntimeError as e:
            assert "trunk mismatch" in str(e), str(e)
        # 逆方向: stem 重みを semantic モデルに読ませても明示エラー
        tmp_stem = tmp_v3 + ".stem.safetensors"
        try:
            save_lllite_model(tmp_stem, lllite_stem, dtype=torch.float32)
            try:
                load_lllite_weights(lllite_v3_b, tmp_stem, strict=False)
                raise AssertionError("expected trunk mismatch error (stem -> semantic)")
            except RuntimeError as e:
                assert "trunk mismatch" in str(e), str(e)
        finally:
            if os.path.exists(tmp_stem):
                os.unlink(tmp_stem)
        logger.info("  v3 trunk mix-up detection OK")
    finally:
        if os.path.exists(tmp_v3):
            os.unlink(tmp_v3)

    # ------------------------------------------------------------------
    # v3 dual: ref_block 2 個の concat trunk
    # ------------------------------------------------------------------
    dit_dual = _DummyDiTV3(num_blocks=4)
    lllite_dual = ControlNetLLLiteDiT(
        dit_dual, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
        cond_dim=64, cond_resblocks=1,
        cond_input_space="latent", trunk="semantic", ref_block="1,3", ref_timestep=0.0,
    )
    assert lllite_dual.ref_blocks == (1, 3)
    assert lllite_dual.ref_blocks_str == "1,3"
    assert lllite_dual.conditioning1.num_ref_blocks == 2
    keys_dual = list(lllite_dual.state_dict().keys())
    assert "conditioning1.ln_in.0.weight" in keys_dual and "conditioning1.ln_in.1.weight" in keys_dual, keys_dual[:8]
    assert "conditioning1.ln_in.weight" not in keys_dual
    assert lllite_dual.conditioning1.proj_in.weight.shape == (64, 2 * MODEL_DIM)
    logger.info("  v3 dual build / keys / proj_in shape OK")

    # dual: set_cond_hidden_states (6-dim) + zero-init forward + K 不一致 assert
    lllite_dual.apply_to()
    h_ref_dual = torch.randn(1, 2, 1, H_v3, W_v3, MODEL_DIM)
    lllite_dual.set_cond_hidden_states(h_ref_dual)
    lllite_dual._update_t_local(t_emb)
    assert lllite_dual.last_cond_hw == (H_v3, W_v3)
    mod_dual = lllite_dual.lllite_modules[0]
    assert mod_dual.cond_emb is not None and mod_dual.cond_emb.shape == (1, H_v3 * W_v3, 32)
    x_dual = torch.randn(1, H_v3 * W_v3, mod_dual.org_module[0].in_features)
    assert torch.allclose(mod_dual(x_dual), mod_dual.org_forward(x_dual)), "v3 dual zero-init forward mismatch"
    lllite_dual.clear_cond_image()
    try:
        lllite_dual.set_cond_hidden_states(h_ref)  # 5-dim (K=1) を dual trunk に渡すと拒否
        raise AssertionError("expected K mismatch assert")
    except AssertionError as e:
        assert "ref block" in str(e), str(e)
    try:
        lllite_v3.set_cond_hidden_states(h_ref_dual)  # 6-dim (K=2) を single trunk に渡すと拒否
        raise AssertionError("expected K mismatch assert (dual -> single)")
    except AssertionError as e:
        assert "ref block" in str(e), str(e)
    lllite_v3.clear_cond_image()
    logger.info("  v3 dual set_cond_hidden_states / zero-init forward / K mismatch asserts OK")

    # dual: save/load round-trip + single/dual 取り違え検出
    tmp_dual = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False).name
    try:
        save_lllite_model(tmp_dual, lllite_dual, dtype=torch.float32, metadata={
            "lllite.version": LLLITE_ARCH_VERSION_SEMANTIC,
            "lllite.trunk": "semantic",
            "lllite.cond_input_space": "latent",
            "lllite.ref_block": lllite_dual.ref_blocks_str,
        })
        lllite_dual_b = ControlNetLLLiteDiT(
            _DummyDiTV3(num_blocks=4), cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
            cond_dim=64, cond_resblocks=1,
            cond_input_space="latent", trunk="semantic", ref_block=(1, 3),
        )
        load_lllite_weights(lllite_dual_b, tmp_dual, strict=True)
        sd_da = lllite_dual.state_dict()
        sd_db = lllite_dual_b.state_dict()
        assert set(sd_da.keys()) == set(sd_db.keys())
        for k in sd_da:
            assert torch.allclose(sd_da[k].float(), sd_db[k].float()), f"dual round-trip mismatch at {k}"
        logger.info("  v3 dual save / load round-trip OK")

        # dual 重みを single モデルへ / single 重みを dual モデルへ → 明示エラー
        lllite_single = ControlNetLLLiteDiT(
            _DummyDiTV3(num_blocks=4), cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_q",
            cond_dim=64, cond_resblocks=1,
            cond_input_space="latent", trunk="semantic", ref_block=2,
        )
        try:
            load_lllite_weights(lllite_single, tmp_dual, strict=False)
            raise AssertionError("expected ref block count mismatch error (dual -> single)")
        except RuntimeError as e:
            assert "ref block count mismatch" in str(e), str(e)
        tmp_single = tmp_dual + ".single.safetensors"
        try:
            save_lllite_model(tmp_single, lllite_single, dtype=torch.float32)
            try:
                load_lllite_weights(lllite_dual_b, tmp_single, strict=False)
                raise AssertionError("expected ref block count mismatch error (single -> dual)")
            except RuntimeError as e:
                assert "ref block count mismatch" in str(e), str(e)
        finally:
            if os.path.exists(tmp_single):
                os.unlink(tmp_single)
        logger.info("  v3 single/dual mix-up detection OK")
    finally:
        if os.path.exists(tmp_dual):
            os.unlink(tmp_dual)

    logger.info("Phase A (v2 + v3 + dual) dummy check PASSED")
