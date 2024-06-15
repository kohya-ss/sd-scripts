# some modules/classes are copied and modified from https://github.com/mcmonkey4eva/sd3-ref 
# the original code is licensed under the MIT License

# and some module/classes are contributed from KohakuBlueleaf. Thanks for the contribution!

from functools import partial
import math
from typing import Dict, Optional
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTokenizer, T5TokenizerFast


memory_efficient_attention = None
try:
    import xformers
except:
    pass

try:
    from xformers.ops import memory_efficient_attention
except:
    memory_efficient_attention = None


# region tokenizer
class SDTokenizer:
    def __init__(
        self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True, min_length=None
    ):
        """
        サブクラスで各種の設定を行ってる。このクラスはその設定に基づき重み付きのトークン化を行うようだ。
        Some settings are done in subclasses. This class seems to perform tokenization with weights based on those settings.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str):
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here.
        The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
        """
        ja: テキストをトークン化し、重み値を持ちます - すべての値に1.0を仮定し、他の機能を無視します。
        詳細は参考実装には関係なく、重み自体はSD3に対して弱い影響しかありません。へぇ～
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(" ")
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)["input_ids"][self.tokens_start : -1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""

    def __init__(self):
        super().__init__(
            pad_with_end=False,
            tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=77,
        )


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class SD3Tokenizer:
    def __init__(self, t5xxl=True):
        # TODO cache tokenizer settings locally or hold them in the repo like ComfyUI
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer() if t5xxl else None

    def tokenize_with_weights(self, text: str):
        return (
            self.clip_l.tokenize_with_weights(text),
            self.clip_g.tokenize_with_weights(text),
            self.t5xxl.tokenize_with_weights(text) if self.t5xxl is not None else None,
        )


# endregion

# region mmdit


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    scaling_factor=None,
    offset=None,
):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(
    embed_dim,
    pos,
    device=None,
    dtype=torch.float32,
):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega *= 2.0 / embed_dim
    omega = 1.0 / 10000**omega
    out = torch.outer(pos.reshape(-1), omega)
    emb = torch.cat([out.sin(), out.cos()], dim=1)
    return emb


def get_2d_sincos_pos_embed_torch(
    embed_dim,
    w,
    h,
    val_center=7.5,
    val_magnitude=7.5,
    device=None,
    dtype=torch.float32,
):
    small = min(h, w)
    val_h = (h / small) * val_magnitude
    val_w = (w / small) * val_magnitude
    grid_h, grid_w = torch.meshgrid(
        torch.linspace(-val_h + val_center, val_h + val_center, h, device=device, dtype=dtype),
        torch.linspace(-val_w + val_center, val_w + val_center, w, device=device, dtype=dtype),
        indexing="ij",
    )
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)  # (H*W, D)
    return emb


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def default(x, default_value):
    if x is None:
        return default_value
    return x


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
    #     device=t.device, dtype=t.dtype
    # )
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(dtype=t.dtype)
    return embedding


def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_channels=3,
        embed_dim=512,
        norm_layer=None,
        flatten=True,
        bias=True,
        strict_img_size=True,
        dynamic_img_pad=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size**2
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)
        self.norm = nn.Identity() if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.dynamic_img_pad:
            # Pad input so we won't have partial patch
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# FinalLayer in mmdit.py
class UnPatch(nn.Module):
    def __init__(self, hidden_size=512, patch_size=4, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.c = out_channels

        # eps is default in mmdit.py
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cmod, H=None, W=None):
        b, n, _ = x.shape
        p = self.patch_size
        c = self.c
        if H is None and W is None:
            w = h = int(n**0.5)
            assert h * w == n
        else:
            h = H // p if H else n // (W // p)
            w = W // p if W else n // h
            assert h * w == n

        shift, scale = self.adaLN_modulation(cmod).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h * p, w * p)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=lambda: nn.GELU(),
        norm_layer=None,
        bias=True,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv = use_conv

        layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = layer(in_features, hidden_features, bias=bias)
        self.fc2 = layer(hidden_features, out_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, t, dtype=None, **kwargs):
        t_freq = timestep_embedding(t, self.freq_embed_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = rmsnorm(x, eps=self.eps)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# Linears for SelfAttention in mmdit.py
class AttentionLinears(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pre_only: bool = False,
        qk_norm: str = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        output:
            q, k, v: [B, L, D]
        """
        B, L, C = x.shape
        qkv: torch.Tensor = self.qkv(x)
        q, k, v = qkv.reshape(B, L, -1, self.head_dim).chunk(3, dim=2)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x


MEMORY_LAYOUTS = {
    "torch": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
        lambda x: (1, x, 1, 1),
    ),
    "xformers": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim),
        lambda x: x.reshape(x.shape[0], x.shape[1], -1),
        lambda x: (1, 1, x, 1),
    ),
    "math": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
        lambda x: (1, x, 1, 1),
    ),
}
# ATTN_FUNCTION = {
#     "torch": F.scaled_dot_product_attention,
#     "xformers": memory_efficient_attention,
# }


def vanilla_attention(q, k, v, mask, scale=None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = einops.rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = einops.repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    return torch.bmm(p_attn, v)


def attention(q, k, v, head_dim, mask=None, scale=None, mode="xformers"):
    """
    q, k, v: [B, L, D]
    """
    pre_attn_layout = MEMORY_LAYOUTS[mode][0]
    post_attn_layout = MEMORY_LAYOUTS[mode][1]
    q = pre_attn_layout(q, head_dim)
    k = pre_attn_layout(k, head_dim)
    v = pre_attn_layout(v, head_dim)

    # scores = ATTN_FUNCTION[mode](q, k.to(q), v.to(q), mask, scale=scale)
    if mode == "torch":
        assert scale is None
        scores = F.scaled_dot_product_attention(q, k.to(q), v.to(q), mask)  # , scale=scale)
    elif mode == "xformers":
        scores = memory_efficient_attention(q, k.to(q), v.to(q), mask, scale=scale)
    else:
        scores = vanilla_attention(q, k.to(q), v.to(q), mask, scale=scale)

    scores = post_attn_layout(scores)
    return scores


class SelfAttention(AttentionLinears):
    def __init__(self, dim, num_heads=8, mode="xformers"):
        super().__init__(dim, num_heads, qkv_bias=True, pre_only=False)
        assert mode in MEMORY_LAYOUTS
        self.head_dim = dim // num_heads
        self.attn_mode = mode

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    def forward(self, x):
        q, k, v = self.pre_attention(x)
        attn_score = attention(q, k, v, self.head_dim, mode=self.attn_mode)
        return self.post_attention(attn_score)


class TransformerBlock(nn.Module):
    def __init__(self, context_size, mode="xformers"):
        super().__init__()
        self.context_size = context_size
        self.norm1 = nn.LayerNorm(context_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(context_size, mode=mode)
        self.norm2 = nn.LayerNorm(context_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(
            in_features=context_size,
            hidden_features=context_size * 4,
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, context_size, num_layers, mode="xformers"):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(context_size, mode) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(context_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# DismantledBlock in mmdit.py
class SingleDiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in MEMORY_LAYOUTS
        self.attn_mode = attn_mode
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionLinears(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            qk_norm=qk_norm,
        )
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = MLP(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=lambda: nn.GELU(approximate="tanh"),
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                )
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not self.pre_only:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(
                    c
                ).chunk(6, dim=-1)
            else:
                shift_msa = None
                shift_mlp = None
                (
                    scale_msa,
                    gate_msa,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(
                    c
                ).chunk(4, dim=-1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (
                x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )
        else:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                ) = self.adaLN_modulation(
                    c
                ).chunk(2, dim=-1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# JointBlock + block_mixing in mmdit.py
class MMDiTBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        self.context_block = SingleDiTBlock(*args, pre_only=pre_only, **kwargs)
        self.x_block = SingleDiTBlock(*args, pre_only=False, **kwargs)
        self.head_dim = self.x_block.attn.head_dim
        self.mode = self.x_block.attn_mode
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _forward(self, context, x, c):
        ctx_qkv, ctx_intermediate = self.context_block.pre_attention(context, c)
        x_qkv, x_intermediate = self.x_block.pre_attention(x, c)

        ctx_len = ctx_qkv[0].size(1)

        q = torch.concat((ctx_qkv[0], x_qkv[0]), dim=1)
        k = torch.concat((ctx_qkv[1], x_qkv[1]), dim=1)
        v = torch.concat((ctx_qkv[2], x_qkv[2]), dim=1)

        attn = attention(q, k, v, head_dim=self.head_dim, mode=self.mode)
        ctx_attn_out = attn[:, :ctx_len]
        x_attn_out = attn[:, ctx_len:]

        x = self.x_block.post_attention(x_attn_out, *x_intermediate)
        if not self.context_block.pre_only:
            context = self.context_block.post_attention(ctx_attn_out, *ctx_intermediate)
        else:
            context = None
        return context, x

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        # hidden_size: Optional[int] = None,
        # num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        use_checkpoint: bool = False,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches=None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        context_processor_layers=None,
        context_size=4096,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.gradient_checkpointing = use_checkpoint

        # hidden_size = default(hidden_size, 64 * depth)
        # num_heads = default(num_heads, hidden_size // 64)

        # apply magic --> this defines a head_size of 64
        self.hidden_size = 64 * depth
        num_heads = depth

        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
        )
        self.t_embedder = TimestepEmbedding(self.hidden_size)

        self.y_embedder = None
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = Embedder(adm_in_channels, self.hidden_size)

        if context_processor_layers is not None:
            self.context_processor = Transformer(context_size, context_processor_layers, attn_mode)
        else:
            self.context_processor = None

        self.context_embedder = nn.Linear(context_size, self.hidden_size)
        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.empty(1, num_patches, self.hidden_size),
            )
        else:
            self.pos_embed = None

        self.use_checkpoint = use_checkpoint
        self.joint_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    self.hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    qkv_bias=qkv_bias,
                    pre_only=i == depth - 1,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        for block in self.joint_blocks:
            block.gradient_checkpointing = use_checkpoint

        self.final_layer = UnPatch(self.hidden_size, patch_size, self.out_channels)
        # self.initialize_weights()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for block in self.joint_blocks:
            block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.joint_blocks:
            block.disable_gradient_checkpointing()

    def initialize_weights(self):
        # TODO: Init context_embedder?
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.pos_embed.shape[-2] ** 0.5),
                scaling_factor=self.pos_embed_scaling_factor,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if getattr(self, "y_embedder", None) is not None:
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def cropped_pos_embed(self, h, w, device=None):
        p = self.x_embedder.patch_size
        # patched size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = self.pos_embed.reshape(
            1,
            self.pos_embed_max_size,
            self.pos_embed_max_size,
            self.pos_embed.shape[-1],
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of class labels
        """

        if self.context_processor is not None:
            context = self.context_processor(context)

        B, C, H, W = x.shape
        x = self.x_embedder(x) + self.cropped_pos_embed(H, W, device=x.device).to(dtype=x.dtype)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        if context is not None:
            context = self.context_embedder(context)

        if self.register_length > 0:
            context = torch.cat(
                (
                    einops.repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    default(context, torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        for block in self.joint_blocks:
            context, x = block(context, x, c)
        x = self.final_layer(x, c, H, W)  # Our final layer combined UnPatchify
        return x[:, :, :H, :W]


def create_mmdit_sd3_medium_configs(attn_mode: str):
    # {'patch_size': 2, 'depth': 24, 'num_patches': 36864,
    # 'pos_embed_max_size': 192, 'adm_in_channels': 2048, 'context_embedder':
    # {'target': 'torch.nn.Linear', 'params': {'in_features': 4096, 'out_features': 1536}}}
    mmdit = MMDiT(
        input_size=None,
        pos_embed_max_size=192,
        patch_size=2,
        in_channels=16,
        adm_in_channels=2048,
        depth=24,
        mlp_ratio=4,
        qk_norm=None,
        num_patches=36864,
        context_size=4096,
        attn_mode=attn_mode,
    )
    return mmdit


# endregion

# region VAE


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device
            )
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(
        self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=16, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        resolution=256,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)


# endregion


# region Text Encoder
class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, mode="xformers"):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.attn_mode = mode

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask, mode=self.attn_mode)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda: (lambda a: a * torch.sigmoid(1.702 * a)),
    # "gelu": torch.nn.functional.gelu,
    "gelu": lambda: nn.GELU(),
}


class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        # # self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        # self.mlp = Mlp(
        #     embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation], dtype=dtype, device=device
        # )
        self.mlp = MLP(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation])
        self.mlp.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in range(num_layers)]
        )

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = torch.nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)

        if x.dtype == torch.bfloat16:
            causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device).fill_(float("-inf")).triu_(1)
            causal_mask = causal_mask.to(dtype=x.dtype)
        else:
            causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)

        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        device="cpu",
        max_length=77,
        layer="last",
        layer_idx=None,
        textmodel_json_config=None,
        dtype=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        return_projected_pooled=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def set_attn_mode(self, mode):
        raise NotImplementedError("This model does not support setting the attention mode")

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = torch.LongTensor(tokens).to(device)
        outputs = self.transformer(
            tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state
        )
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output

    def set_attn_mode(self, mode):
        clip_text_model = self.transformer.text_model
        for layer in clip_text_model.encoder.layers:
            layer.self_attn.set_attn_mode(mode)


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )

    def set_attn_mode(self, mode):
        clip_text_model = self.transformer.text_model
        for layer in clip_text_model.encoder.layers:
            layer.self_attn.set_attn_mode(mode)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""

    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=T5,
        )

    def set_attn_mode(self, mode):
        t5: T5 = self.transformer
        for t5block in t5.encoder.block:
            t5block: T5Block
            t5layer: T5LayerSelfAttention = t5block.layer[0]
            t5SaSa: T5Attention = t5layer.SelfAttention
            t5SaSa.set_attn_mode(mode)


#################################################################################################
### T5 implementation, for the T5-XXL text encoder portion, largely pulled from upstream impl
#################################################################################################


class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""

    def __init__(self):
        super().__init__(
            pad_with_end=False,
            tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=77,
        )


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=x.device, dtype=x.dtype) * x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device)

        self.attn_mode = "xformers"  # TODO 何とかする

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        out = attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask, mode=self.attn_mode)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, vocab_size, dtype, device):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList(
            [
                T5Block(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias=(i == 0), dtype=dtype, device=device)
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.embed_tokens(input_ids)
        past_bias = None
        for i, l in enumerate(self.block):
            # print(i, x.mean(), x.std())
            x, past_bias = l(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        # print(x.mean(), x.std())
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        # print(x.mean(), x.std())
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        self.encoder = T5Stack(
            self.num_layers,
            config_dict["d_model"],
            config_dict["d_model"],
            config_dict["d_ff"],
            config_dict["num_heads"],
            config_dict["vocab_size"],
            dtype,
            device,
        )
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


def create_clip_l(device="cpu", dtype=torch.float32, state_dict: Optional[Dict[str, torch.Tensor]] = None):
    r"""
    state_dict is not loaded, but updated with missing keys
    """
    CLIPL_CONFIG = {
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
    }
    with torch.no_grad():
        clip_l = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device=device,
            dtype=dtype,
            layer_norm_hidden_state=False,
            return_projected_pooled=False,
            textmodel_json_config=CLIPL_CONFIG,
        )
    if state_dict is not None:
        # update state_dict if provided to include logit_scale and text_projection.weight avoid errors
        if "logit_scale" not in state_dict:
            state_dict["logit_scale"] = clip_l.logit_scale
        if "transformer.text_projection.weight" not in state_dict:
            state_dict["transformer.text_projection.weight"] = clip_l.transformer.text_projection.weight
    return clip_l


def create_clip_g(device="cpu", dtype=torch.float32, state_dict: Optional[Dict[str, torch.Tensor]] = None):
    r"""
    state_dict is not loaded, but updated with missing keys
    """
    CLIPG_CONFIG = {
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
    }
    with torch.no_grad():
        clip_g = SDXLClipG(CLIPG_CONFIG, device=device, dtype=dtype)
    if state_dict is not None:
        if "logit_scale" not in state_dict:
            state_dict["logit_scale"] = clip_g.logit_scale
    return clip_g


def create_t5xxl(device="cpu", dtype=torch.float32, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> T5XXLModel:
    T5_CONFIG = {"d_ff": 10240, "d_model": 4096, "num_heads": 64, "num_layers": 24, "vocab_size": 32128}
    with torch.no_grad():
        t5 = T5XXLModel(T5_CONFIG, dtype=dtype, device=device)
    if state_dict is not None:
        if "logit_scale" not in state_dict:
            state_dict["logit_scale"] = t5.logit_scale
        if "transformer.shared.weight" in state_dict:
            state_dict.pop("transformer.shared.weight")
    return t5


# endregion
