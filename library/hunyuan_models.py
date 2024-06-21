import math
from typing import Tuple, Union, Optional, Any

import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import einops
from einops import repeat

from diffusers import AutoencoderKL
from timm.models.vision_transformer import Mlp
from timm.models.layers import to_2tuple
from transformers import (
    AutoTokenizer,
    MT5EncoderModel,
    BertModel,
)

memory_efficient_attention = None
try:
    import xformers
except:
    pass

try:
    from xformers.ops import memory_efficient_attention
except:
    memory_efficient_attention = None


VAE_SCALE_FACTOR = 0.13025
MODEL_VERSION_HUNYUAN_V1_1 = "HunyuanDiT-v1.1"


class MT5Embedder(nn.Module):
    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        torch_dtype=None,
        use_tokenizer_only=False,
        max_length=128,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                # "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
        model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if use_tokenizer_only:
            return
        self.model = (
            MT5EncoderModel.from_pretrained(model_dir, **model_kwargs)
            .eval()
            .to(self.torch_dtype)
        )

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        outputs = self.model(
            input_ids=text_tokens_and_mask["input_ids"].to(self.device),
            attention_mask=(
                text_tokens_and_mask["attention_mask"].to(self.device)
                if attention_mask
                else None
            ),
            output_hidden_states=True,
        )
        text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    def get_input_ids(self, caption):
        return self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids

    def get_hidden_states(self, input_ids, layer_index=-1):
        mask = (input_ids != 0).long()
        outputs = self.model(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True
        )
        return outputs["hidden_states"][layer_index], mask


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: Optional[torch.Tensor],
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Precomputed frequency tensor for complex exponentials.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        if xk is not None:
            xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        if xk is not None:
            xk_ = torch.view_as_complex(
                xk.float().reshape(*xk.shape[:-1], -1, 2)
            )  # [B, S, H, D//2]
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


MEMORY_LAYOUTS = {
    "torch": (
        lambda x, head_dim: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
        lambda x: (1, x, 1, 1),
    ),
    "xformers": (
        lambda x, head_dim: x,
        lambda x: x,
        lambda x: (1, 1, x, 1),
    ),
    "math": (
        lambda x, head_dim: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
        lambda x: (1, x, 1, 1),
    ),
}


def vanilla_attention(q, k, v, mask, dropout_p, scale=None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = einops.rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = einops.repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    if dropout_p != 0:
        scores = F.dropout(p_attn, p=dropout_p, training=True)
    return torch.bmm(p_attn, v)


def attention(q, k, v, head_dim, dropout_p=0, mask=None, scale=None, mode="xformers"):
    """
    q, k, v: [B, L, H, D]
    """
    pre_attn_layout = MEMORY_LAYOUTS[mode][0]
    post_attn_layout = MEMORY_LAYOUTS[mode][1]
    q = pre_attn_layout(q, head_dim)
    k = pre_attn_layout(k, head_dim)
    v = pre_attn_layout(v, head_dim)

    # scores = ATTN_FUNCTION[mode](q, k.to(q), v.to(q), mask, scale=scale)
    if mode == "torch":
        assert scale is None
        scores = F.scaled_dot_product_attention(
            q, k.to(q), v.to(q), mask, dropout_p
        )  # , scale=scale)
    elif mode == "xformers":
        scores = memory_efficient_attention(
            q, k.to(q), v.to(q), mask, dropout_p, scale=scale
        )
    else:
        scores = vanilla_attention(q, k.to(q), v.to(q), mask, dropout_p, scale=scale)

    scores = post_attn_layout(scores)
    return scores


class SelfAttention(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
        attn_mode="xformers",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.attn_mode = attn_mode

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    def forward(self, x, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q).to(q)  # [b, s, h, d]
        k = self.k_norm(k).to(q)

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis_img)
            assert (
                qq.shape == q.shape and kk.shape == k.shape
            ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
            q, k = qq, kk

        # qkv = torch.stack([q, k, v], dim=2)  # [b, s, 3, h, d]
        # context = self.inner_attn(qkv)
        context = attention(q, k, v, self.head_dim, self.attn_drop, mode=self.attn_mode)
        out = self.out_proj(context.reshape(b, s, d))
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class CrossAttention(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
        attn_mode="xformers",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )

        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.attn_mode = attn_mode

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    def forward(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num_heads * head_dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // num_heads), RoPE for image
        """
        b, s1, _ = x.shape  # [b, s1, D]
        _, s2, _ = y.shape  # [b, s2, 1024]

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        kv = self.kv_proj(y).view(
            b, s2, 2, self.num_heads, self.head_dim
        )  # [b, s2, 2, h, d]
        k, v = kv.unbind(dim=2)  # [b, s2, h, d]
        q = self.q_norm(q).to(q)  # [b, s1, h, d]
        k = self.k_norm(k).to(k)  # [b, s2, h, d]

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            assert qq.shape == q.shape, f"qq: {qq.shape}, q: {q.shape}"
            q = qq  # [b, s1, h, d]
        # kv = torch.stack([k, v], dim=2)  # [b, s1, 2, h, d]
        # context = self.inner_attn(q, kv)  # [b, s1, h, d]
        context = attention(q, k, v, self.head_dim, self.attn_drop, mode=self.attn_mode)
        context = context.reshape(b, s1, -1)  # [b, s1, D]

        out = self.out_proj(context)
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_size = tuple(img_size)
        else:
            raise ValueError(
                f"img_size must be int or tuple/list of length 2. Got {img_size}"
            )
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def update_image_size(self, img_size):
        self.img_size = img_size
        self.grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def timestep_embedding(t, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            (-math.log(max_period) / half)
            * torch.arange(start=0, end=half, dtype=torch.float32)
        ).to(
            device=t.device
        )  # size: [dim/2], 一个指数衰减的曲线
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = t.unsqueeze(-1).repeat(1, dim)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class AttentionPool(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FP32_Layernorm(nn.LayerNorm):
    enable_fp32 = True
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.enable_fp32:
            return F.layer_norm(
                inputs.float(),
                self.normalized_shape,
                self.weight.float(),
                self.bias.float(),
                self.eps,
            ).to(inputs.dtype)
        else:
            return F.layer_norm(
                inputs, self.normalized_shape, self.weight, self.bias, self.eps
            )


class FP32_SiLU(nn.SiLU):
    enable_fp32 = True
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.enable_fp32:
            return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)
        return torch.nn.functional.silu(inputs, inplace=False).to(inputs.dtype)


class HunYuanDiTBlock(nn.Module):
    """
    A HunYuanDiT block with `add` conditioning.
    """

    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        mlp_ratio=4.0,
        text_states_dim=1024,
        qk_norm=False,
        norm_type="layer",
        skip=False,
        attn_mode="xformers",
    ):
        super().__init__()
        self.attn_mode = attn_mode
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )
        self.attn1 = SelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            attn_mode=attn_mode,
        )

        # ========================= FFN =========================
        self.norm2 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            FP32_SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(
            hidden_size,
            text_states_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            attn_mode=attn_mode,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(
                2 * hidden_size, elementwise_affine=True, eps=1e-6
            )
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.gradient_checkpointing = False

    def set_attn_mode(self, attn_mode):
        self.attn1.set_attn_mode(attn_mode)
        self.attn2.set_attn_mode(attn_mode)

    def _forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = (
            self.norm1(x) + shift_msa,
            freq_cis_img,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (self.norm3(x), text_states, freq_cis_img)
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x

    def forward(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            final_hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            final_hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            FP32_SiLU(), nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunYuanDiT(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    def __init__(
        self,
        qk_norm=True,
        norm="layer",
        text_states_dim=1024,
        text_len=77,
        text_states_dim_t5=2048,
        text_len_t5=256,
        learn_sigma=True,
        input_size=(32, 32),
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        log_fn=print,
        attn_mode="xformers",
    ):
        super().__init__()
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5
        self.norm = norm

        log_fn(f"    Use {attn_mode} attention implementation.")
        qk_norm = qk_norm  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(
                self.text_len + self.text_len_t5,
                self.text_states_dim,
                dtype=torch.float32,
            )
        )

        # Attention pooling
        self.pooler = AttentionPool(
            self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=1024
        )

        # Here we use a default learned embedder layer for future extension.
        self.style_embedder = nn.Embedding(1, hidden_size)

        # Image size and crop size conditions
        self.extra_in_dim = 256 * 6 + hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_in_dim += 1024
        self.extra_embedder = nn.Sequential(
            nn.Linear(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.text_states_dim,
                    qk_norm=qk_norm,
                    norm_type=self.norm,
                    skip=layer > depth // 2,
                    attn_mode=attn_mode,
                )
                for layer in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size, hidden_size, patch_size, self.out_channels
        )
        self.unpatchify_channels = self.out_channels

        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = False

    def set_attn_mode(self, attn_mode):
        for block in self.blocks:
            block.set_attn_mode(attn_mode)

    def enable_fp32_layer_norm(self):
        FP32_Layernorm.enable_fp32 = True

    def disable_fp32_layer_norm(self):
        FP32_Layernorm.enable_fp32 = False

    def enable_fp32_silu(self):
        FP32_SiLU.enable_fp32 = True

    def disable_fp32_silu(self):
        FP32_SiLU.enable_fp32 = False

    def forward(
        self,
        x,
        t,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
    ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        """
        # MODIFIED BEGIN
        text_states = encoder_hidden_states  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.bool()  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()  # 2,256
        b_cl, l_cl, c_cl = text_states.shape
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states_t5 = text_states_t5.view(b_t5, l_t5, -1)

        # Support for "CLIP Concat" trick
        # We allow user to use multiple CLIP embed concat together
        # Which means the learanble pad for CLIP only apply on last 76 tokens
        padding = self.text_embedding_padding.to(text_states)
        padding_clip = torch.concat(
            [padding[:1], padding[1:76].repeat((l_cl - 2) // 75, 1), padding[76:77]],
            dim=0,
        )
        text_states = torch.where(
            text_states_mask.unsqueeze(2),
            text_states,
            padding_clip,
        )
        text_states_t5 = torch.where(
            text_states_t5_mask.unsqueeze(2),
            text_states_t5,
            padding[77:],
        )
        text_states = torch.cat([text_states, text_states_t5], dim=1)
        # MODIFIED END

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Build image meta size tokens
        image_meta_size = timestep_embedding(
            image_meta_size.view(-1), 256
        )  # [B * 6, 256]
        image_meta_size = image_meta_size.to(x)
        image_meta_size = image_meta_size.view(-1, 6 * 256)
        extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        # Build style tokens
        style_embedding = self.style_embedder(style)
        extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                skip = skips.pop()
                x = block(x, c, text_states, freqs_cis_img, skip)  # (N, L, D)
            else:
                x = block(x, c, text_states, freqs_cis_img)  # (N, L, D)

            if layer < (self.depth // 2 - 1):
                skips.append(x)
        # ========================= Final layer =========================
        x = self.final_layer(x, c)  # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)  # (N, out_channels, H, W)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.extra_embedder[0].weight, std=0.02)
        nn.init.normal_(self.extra_embedder[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in HunYuanDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.default_modulation[-1].weight, 0)
            nn.init.constant_(block.default_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


#################################################################################
#                            HunYuanDiT Configs                                 #
#################################################################################

HUNYUAN_DIT_CONFIG = {
    "DiT-g/2": {
        "depth": 40,
        "hidden_size": 1408,
        "patch_size": 2,
        "num_heads": 16,
        "mlp_ratio": 4.3637,
    },
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
}


def DiT_g_2(**kwargs):
    return (
        HunYuanDiT(
            depth=40,
            hidden_size=1408,
            patch_size=2,
            num_heads=16,
            mlp_ratio=4.3637,
            **kwargs,
        ),
        2,
        88,
    )


def DiT_XL_2(**kwargs):
    return (
        HunYuanDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
        2,
        72,
    )


HUNYUAN_DIT_MODELS = {
    "DiT-g/2": DiT_g_2,
    "DiT-XL/2": DiT_XL_2,
}


if __name__ == "__main__":
    denoiser: HunYuanDiT = DiT_g_2(input_size=(128, 128))
    sd = torch.load("./model/denoiser/pytorch_model_module.pt")
    denoiser.load_state_dict(sd)
    denoiser.half().cuda()
    denoiser.enable_gradient_checkpointing()

    clip_tokenizer = AutoTokenizer.from_pretrained("./model/clip")
    clip_encoder = BertModel.from_pretrained("./model/clip").half().cuda()

    mt5_embedder = MT5Embedder("./model/mt5", torch_dtype=torch.float16, max_length=256)

    vae = AutoencoderKL.from_pretrained("./model/vae").half().cuda()

    print(sum(p.numel() for p in denoiser.parameters()) / 1e6)
    print(sum(p.numel() for p in mt5_embedder.parameters()) / 1e6)
    print(sum(p.numel() for p in clip_encoder.parameters()) / 1e6)
    print(sum(p.numel() for p in vae.parameters()) / 1e6)
