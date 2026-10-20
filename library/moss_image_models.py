# MOSS-Image DiT model
# Original implementation: OpenMOSS (pipeline_moss_image.py in OpenMOSS-Team/MOSS-Image-Base), licensed under Apache-2.0.
#
# Differences from the original implementation:
# - Batched forward: the original loops over samples (each sample runs at its true text length). This implementation
#   right-pads the text tokens and uses attention masks / variable-length attention (library.attention) instead.
#   The RoPE is built per sample so that image tokens use the true text length of each sample for the "f" axis.
# - Main-block projections are plain attributes (q, k, v, o, ffn_up_gate, ffn_down) of a small nn.Module instead of
#   nn.ModuleDict. The state dict keys are identical to the official checkpoint.
# - Gradient checkpointing (optionally with CPU / unsloth offloading), block swapping and per-block torch.compile hooks.

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import custom_offloading_utils
from library.attention import AttentionParams, attention
from library.custom_offloading_utils import to_cpu, to_device, unsloth_checkpoint
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region configuration

MOSS_IMAGE_CONFIG = {
    "dim": 3072,
    "ffn_dim": 8192,
    "num_heads": 24,
    "num_layers": 42,
    "text_dim": 4096,
    "freq_dim": 256,
    "latent_channels": 128,
    "rope_base": 10000.0,
}

# endregion


# region building blocks


class MossImageRMSNorm(nn.Module):
    """RMSNorm that normalizes in float32 and applies the weight in the input dtype (matches the official implementation)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def situ_glu(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SiTU-GLU activation used by MOSS-Image FFNs."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding.

    Args:
        x: [B, L, H, D]
        rope: [B, L, D] (first half: sin, second half: cos) or [1, L, D]
    """
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)  # [B, L, 1, D]
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def moss_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: torch.Tensor, attn_params: AttentionParams
) -> torch.Tensor:
    """Rotate q/k, run attention and flatten heads.

    Args:
        q, k, v: [B, L, H, D]
        rope: [B, L, D]
    Returns:
        [B, L, H*D]
    """
    q = apply_rope(q, rope).to(v.dtype)
    k = apply_rope(k, rope).to(v.dtype)
    qkv = [q, k, v]
    del q, k, v
    return attention(qkv, attn_params=attn_params)


class MossImageTextBlock(nn.Module):
    """Transformer block of the text refiner (no modulation, fused qkv, separate ffn_up / ffn_gate, with bias)."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = MossImageRMSNorm(dim)
        self.norm2 = MossImageRMSNorm(dim)
        self.q_norm = MossImageRMSNorm(dim // heads)
        self.k_norm = MossImageRMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def _forward(self, x: torch.Tensor, rope: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1)))
        k = self.k_norm(k.unflatten(-1, (self.heads, -1)))
        v = v.unflatten(-1, (self.heads, -1))
        x = x + self.proj(moss_attention(q, k, v, rope, attn_params))
        y = self.norm2(x)
        return x + self.ffn_down(situ_glu(self.ffn_up(y), self.ffn_gate(y)))

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        return _checkpointed_forward(self, self._forward, x, rope, attn_params)


class MossImageTextEmbedder(nn.Module):
    """Projects text encoder hidden states to the DiT dimension and refines them with two transformer blocks."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int, num_blocks: int = 2):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossImageTextBlock(dim, ffn_dim, heads) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, attn_params)
        return x


class MossImageBlockProj(nn.Module):
    """Projections of a main block. The attribute names match the official state dict keys (`layers.N.proj.q.weight` etc.)."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ffn_up_gate = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)


class MossImageBlock(nn.Module):
    """Main single-stream transformer block with shared (per-timestep) modulation."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = MossImageRMSNorm(dim)
        self.norm2 = MossImageRMSNorm(dim)
        self.q_norm = MossImageRMSNorm(dim // heads)
        self.k_norm = MossImageRMSNorm(dim // heads)
        self.proj = MossImageBlockProj(dim, ffn_dim)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def _forward(
        self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor, attn_params: AttentionParams
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N + L, D] image tokens followed by (right-padded) text tokens
            rope: [B, N + L, head_dim]
            modulation: [B, 6, D]
            attn_params: attention parameters built from the text mask (img_len = N)
        """
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        # compute dtype of the block: the norm weight is always floating point (the projections may be ConvRot INT8)
        dtype = self.norm1.weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj.q(y).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(self.proj.k(y).unflatten(-1, (self.heads, -1)))
        v = self.proj.v(y).unflatten(-1, (self.heads, -1))
        del y
        x = x + self.proj.o(moss_attention(q, k, v, rope, attn_params)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj.ffn_up_gate(y.to(dtype)).chunk(2, -1)
        del y
        return x + self.proj.ffn_down(situ_glu(hidden, gate)) * gate_f[:, None]

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor, attn_params: AttentionParams
    ) -> torch.Tensor:
        return _checkpointed_forward(self, self._forward, x, rope, modulation, attn_params)


def _checkpointed_forward(module: nn.Module, forward_fn, *args):
    """Run forward_fn with the gradient checkpointing mode configured on the module."""
    if not (module.training and module.gradient_checkpointing):
        return forward_fn(*args)

    if module.unsloth_offload_checkpointing:
        # Unsloth: async non-blocking CPU RAM offload
        return unsloth_checkpoint(forward_fn, *args)
    elif module.cpu_offload_checkpointing:
        # Standard cpu offload: blocking transfers
        def create_custom_forward(func):
            def custom_forward(*inputs):
                device = next(t.device for t in inputs if isinstance(t, torch.Tensor))
                outputs = func(*to_device(inputs, device))
                return to_cpu(outputs)

            return custom_forward

        return torch_checkpoint(create_custom_forward(forward_fn), *args, use_reentrant=False)
    else:
        return torch_checkpoint(forward_fn, *args, use_reentrant=False)


class MossImageRoPE(nn.Module):
    """4D rotary embedding with coordinates (r, f, h, w).

    Text tokens: (0, j, 0, 0) for j in [0, L). Image tokens: (0, L, h - (H-1)/2, w - (W-1)/2) where L is the number of
    valid text tokens of the sample. The sequence order is [image, text].
    """

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        # `bands` is a derived constant. It is kept in the state dict for checkpoint compatibility, but the value
        # is always recomputed (see `_default_bands`), including after loading or dtype casting.
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device: torch.device) -> torch.Tensor:
        # Match the official implementation: CPU fp32 construction, then move to the device.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def forward(self, height: int, width: int, text_length: int, text_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            height, width: image token grid size
            text_length: padded text length L_max
            text_lens: [B] number of valid text tokens per sample (int tensor)
        Returns:
            [B, H*W + L_max, head_dim] float32 (first half sin, second half cos)
        """
        device = text_lens.device
        if self.bands.dtype != torch.float32 or self.bands.device != device:
            self.bands = self._default_bands(device)

        batch = text_lens.shape[0]
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()  # [N, 4]
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        image = image.unsqueeze(0).repeat(batch, 1, 1)  # [B, N, 4]
        image[:, :, 1] = text_lens.to(torch.float32)[:, None]  # f axis = true text length of the sample

        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        text = text.unsqueeze(0).expand(batch, -1, -1)  # [B, L, 4]

        phase = torch.cat((image, text), 1).unsqueeze(-1) * self.bands  # [B, N + L, 4, num_bands]
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)  # [B, N + L, head_dim]


# endregion


# region model


class MossImage(nn.Module):
    """MOSS-Image DiT (single-stream, 42 blocks) with training-compatible state dict keys."""

    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MossImageTextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.rope = MossImageRoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossImageBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = MossImageRMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        # block swap
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

    @property
    def device(self) -> torch.device:
        return self.video_embedder.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.video_embedder.weight.dtype

    # gradient checkpointing

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        for block in self.layers:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)
        for block in self.text_embedder.blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)
        logger.info(f"MossImage: Gradient checkpointing enabled. cpu_offload: {cpu_offload}, unsloth_offload: {unsloth_offload}")

    def disable_gradient_checkpointing(self):
        for block in self.layers:
            block.disable_gradient_checkpointing()
        for block in self.text_embedder.blocks:
            block.disable_gradient_checkpointing()
        logger.info("MossImage: Gradient checkpointing disabled.")

    # block swap

    def enable_block_swap(self, num_blocks: int, config: custom_offloading_utils.BlockSwapConfig):
        self.blocks_to_swap = num_blocks
        assert (
            self.blocks_to_swap <= self.num_layers - 2
        ), f"Cannot swap more than {self.num_layers - 2} blocks. Requested: {self.blocks_to_swap} blocks."

        self.offloader = custom_offloading_utils.create_offloader("layer", self.layers, self.num_layers, self.blocks_to_swap, config)
        logger.info(
            f"MossImage: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_layers}, device: {config.device}."
            f" Supports backward: {config.supports_backward}. H2D-only: {config.h2d_only}"
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # Move all modules to device except the swapped blocks (which are managed by the offloader)
        if self.blocks_to_swap:
            save_layers = self.layers
            self.layers = None

        self.to(device)

        if self.blocks_to_swap:
            self.layers = save_layers

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()
        logger.info("MossImage: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()
        logger.info("MossImage: Block swap set to forward and backward.")

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.layers)

    # forward

    def get_time_modulation(self, timesteps: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the shared block modulation [B, 6, D] and the head modulation [B, 2, D].

        Args:
            timesteps: [B] or scalar, sigma * 1000
        """
        times = torch.as_tensor(timesteps, device=self.device, dtype=torch.float32).reshape(-1).expand(batch)
        # The original computes the sinusoid in float64 and feeds float32 into the MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, latent_channels, H, W] patchified and normalized latents
            timesteps: [B] sigma * 1000 (note: not divided by 1000)
            context: [B, L, text_dim] right-padded text encoder hidden states
            context_mask: [B, L] 1 for valid tokens. None means all tokens are valid.
            skip_layers: indices of the main blocks to skip (used for the unconditional branch of CFG)
        Returns:
            [B, latent_channels, H, W]
        """
        batch, _, height, width = x.shape
        if x.shape[1] != self.latent_channels:
            raise ValueError(f"x must be [B, {self.latent_channels}, H, W], got {tuple(x.shape)}")
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("context must be [B, L, text_dim] with the same batch size as x")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.int32, device=x.device)
        context_mask = context_mask.to(torch.int32)
        text_lens = context_mask.sum(1)  # [B]
        if torch.any(text_lens < 1):
            raise ValueError("Each sample needs at least one valid text token.")
        skip_layers = frozenset(skip_layers) if skip_layers else frozenset()

        mod, head = self.get_time_modulation(timesteps, batch)

        img_len = height * width
        text_len = context.shape[1]
        video = x.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()  # [B, N, C]
        video = self.video_embedder(video)
        rope = self.rope(height, width, text_len, text_lens).to(video.dtype)  # [B, N + L, head_dim]

        txt_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, context_mask)
        text = self.text_embedder(context, rope[:, img_len:], txt_attn_params)
        hidden = torch.cat((video, text), 1)
        del video, text

        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, img_len, context_mask)

        for layer_index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(layer_index)

            if layer_index not in skip_layers:
                hidden = layer(hidden, rope, mod, attn_params)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, layer_index)

        hidden = self.final_norm_video(hidden[:, :img_len].to(self.final_norm_video.weight.dtype))
        hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
        hidden = self.final_linear_video(hidden.to(self.final_linear_video.weight.dtype))
        return hidden.transpose(1, 2).reshape(batch, -1, height, width)


# endregion
