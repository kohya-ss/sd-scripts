# Qwen-Image VAE 2D image-only implementation.
#
# This module loads the official Qwen-Image VAE weights by reducing the causal
# Conv3d weights to equivalent Conv2d weights for single-frame image use.

import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.qwen_image_autoencoder_kl import (
    ChunkedConv2d,
    DiagonalGaussianDistribution,
    QwenImageRMS_norm,
    QwenImageUpsample,
    convert_comfyui_state_dict,
)
from library.safetensors_utils import load_safetensors
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


SCALE_FACTOR = 8


class QwenImageResidualBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = QwenImageRMS_norm(in_dim, images=True)
        self.conv1 = ChunkedConv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ChunkedConv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = ChunkedConv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class QwenImageAttentionBlock2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = QwenImageRMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, height, width = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size, channels, height, width)
        x = self.proj(x)
        return x + identity


class QwenImageResample2D(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ChunkedConv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), ChunkedConv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class QwenImageMidBlock2D(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1) -> None:
        super().__init__()
        resnets = [QwenImageResidualBlock2D(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock2D(dim))
            resnets.append(QwenImageResidualBlock2D(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class QwenImageEncoder2D(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        input_channels: int = 3,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float, ...] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_mult = list(dim_mult)
        self.attn_scales = list(attn_scales)

        dims = [dim * multiplier for multiplier in [1] + self.dim_mult]
        scale = 1.0

        self.conv_in = ChunkedConv2d(input_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock2D(in_dim, out_dim, dropout))
                if scale in self.attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock2D(out_dim))
                in_dim = out_dim
            if index != len(self.dim_mult) - 1:
                self.down_blocks.append(QwenImageResample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock2D(out_dim, dropout, num_layers=1)
        self.norm_out = QwenImageRMS_norm(out_dim, images=True)
        self.conv_out = ChunkedConv2d(out_dim, z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class QwenImageUpBlock2D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock2D(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample2D(out_dim, mode=upsample_mode)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageDecoder2D(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        output_channels: int = 3,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float, ...] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_mult = list(dim_mult)

        dims = [dim * multiplier for multiplier in [self.dim_mult[-1]] + self.dim_mult[::-1]]

        self.conv_in = ChunkedConv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock2D(dims[0], dropout, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if index > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if index != len(self.dim_mult) - 1 else None
            self.up_blocks.append(
                QwenImageUpBlock2D(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            )

        self.norm_out = QwenImageRMS_norm(out_dim, images=True)
        self.conv_out = ChunkedConv2d(out_dim, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class AutoencoderKLQwenImage2D(nn.Module):
    _supports_gradient_checkpointing = False

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        input_channels: int = 3,
        spatial_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        self.encoder = QwenImageEncoder2D(
            dim=base_dim,
            z_dim=z_dim * 2,
            input_channels=input_channels,
            dim_mult=tuple(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=tuple(attn_scales),
            dropout=dropout,
        )
        self.quant_conv = ChunkedConv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = ChunkedConv2d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder2D(
            dim=base_dim,
            z_dim=z_dim,
            output_channels=input_channels,
            dim_mult=tuple(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=tuple(attn_scales),
            dropout=dropout,
        )

        self.spatial_chunk_size = None
        if spatial_chunk_size is not None and spatial_chunk_size > 0:
            self.enable_spatial_chunking(spatial_chunk_size)

    @property
    def dtype(self):
        return self.encoder.parameters().__next__().dtype

    @property
    def device(self):
        return self.encoder.parameters().__next__().device

    def enable_spatial_chunking(self, spatial_chunk_size: int) -> None:
        if spatial_chunk_size is None or spatial_chunk_size <= 0:
            raise ValueError(f"`spatial_chunk_size` must be a positive integer, got {spatial_chunk_size}.")
        self.spatial_chunk_size = int(spatial_chunk_size)
        for module in self.modules():
            if isinstance(module, ChunkedConv2d):
                module.spatial_chunk_size = self.spatial_chunk_size

    def disable_spatial_chunking(self) -> None:
        self.spatial_chunk_size = None
        for module in self.modules():
            if isinstance(module, ChunkedConv2d):
                module.spatial_chunk_size = None

    def enable_tiling(self, *args, **kwargs) -> None:
        raise NotImplementedError("Tiling is not implemented for QwenImage 2D VAE. Use spatial chunking instead.")

    def disable_tiling(self) -> None:
        return

    def enable_slicing(self) -> None:
        return

    def disable_slicing(self) -> None:
        return

    def disable_cache(self) -> None:
        return

    def clear_cache(self) -> None:
        return

    def _flatten_frames(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if x.dim() == 4:
            return x, None
        if x.dim() != 5:
            raise ValueError(f"Unsupported QwenImage 2D VAE input shape: {tuple(x.shape)}")

        batch, channels, frames, height, width = x.shape
        if frames != 1:
            raise ValueError(
                f"QwenImage 2D VAE is image-only and does not support T>1 inputs (got shape {tuple(x.shape)})."
            )
        x = x.squeeze(2)
        return x, (batch, frames)

    def _restore_frames(self, x: torch.Tensor, frame_info: Optional[Tuple[int, int]]) -> torch.Tensor:
        if frame_info is None:
            return x

        batch, frames = frame_info
        channels, height, width = x.shape[1:]
        return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x, frame_info = self._flatten_frames(x)
        moments = self.quant_conv(self.encoder(x))
        return self._restore_frames(moments, frame_info)

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[DiagonalGaussianDistribution]]:
        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return {"latent_dist": posterior}

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z, frame_info = self._flatten_frames(z)
        sample = self.decoder(self.post_quant_conv(z))
        sample = torch.clamp(sample, min=-1.0, max=1.0)
        return self._restore_frames(sample, frame_info)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return {"sample": decoded}

    def _latent_stats(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_shape = [1, self.z_dim] + [1] * (latents.dim() - 2)
        latents_mean = torch.tensor(self.latents_mean).view(*stats_shape).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.latents_std).view(*stats_shape).to(latents.device, latents.dtype)
        return latents_mean, latents_std

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.dtype)
        latents_mean, latents_std = self._latent_stats(latents)
        latents = latents / latents_std + latents_mean
        image = self.decode(latents, return_dict=False)[0]
        return image.clamp(-1.0, 1.0)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(self.dtype)
        posterior = self.encode(pixels, return_dict=False)[0]
        latents = posterior.mode()
        latents_mean, latents_std = self._latent_stats(latents)
        return (latents - latents_mean) * latents_std

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        posterior = self.encode(sample, return_dict=False)[0]
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)


def convert_3d_state_dict_to_2d(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert official Qwen-Image causal Conv3d VAE weights for image-only 2D use."""
    state_dict = convert_comfyui_state_dict(sd)
    new_state_dict = {}
    skipped_time_conv = []

    for key, tensor in state_dict.items():
        if ".time_conv." in key:
            skipped_time_conv.append(key)
            continue

        # NOTE: indexing returns a view that shares storage with the source 3D tensor. Since load_vae uses
        # `assign=True`, assigning the view as-is would keep the full 3D weights resident. Call `.contiguous()`
        # so the original 3D tensors can be freed (matters on low-VRAM environments).
        if tensor.ndim == 5:
            tensor = tensor[:, :, -1, :, :].contiguous()
        elif tensor.ndim == 4 and key.endswith(".gamma"):
            tensor = tensor[:, :, :, 0].contiguous()

        new_state_dict[key] = tensor

    if skipped_time_conv:
        logger.info(f"Skipped {len(skipped_time_conv)} temporal-only VAE weights for QwenImage 2D VAE")

    return new_state_dict


def load_vae(
    vae_path: str,
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    spatial_chunk_size: Optional[int] = None,
    disable_cache: bool = False,
) -> AutoencoderKLQwenImage2D:
    """Load the official Qwen-Image VAE as an image-only 2D VAE.

    Note: ``disable_cache`` is accepted only to keep the same signature as the 3D ``load_vae`` (so callers such as
    ``anima_train_utils.load_qwen_image_vae`` can dispatch uniformly). The 2D VAE has no temporal cache, so it is ignored.
    """
    VAE_CONFIG_JSON = """
{
  "_class_name": "AutoencoderKLQwenImage",
  "_diffusers_version": "0.34.0.dev0",
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [
    1,
    2,
    4,
    4
  ],
  "dropout": 0.0,
  "latents_mean": [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921
  ],
  "latents_std": [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916
  ],
  "num_res_blocks": 2,
  "temperal_downsample": [
    false,
    true,
    true
  ],
  "z_dim": 16
}
"""
    if spatial_chunk_size is not None and spatial_chunk_size % 2 != 0:
        spatial_chunk_size += 1
        logger.warning(f"Adjusted spatial_chunk_size to the next even number: {spatial_chunk_size}")

    config = json.loads(VAE_CONFIG_JSON)
    logger.info("Initializing image-only QwenImage 2D VAE")
    vae = AutoencoderKLQwenImage2D(
        base_dim=config["base_dim"],
        z_dim=config["z_dim"],
        dim_mult=tuple(config["dim_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attn_scales=config["attn_scales"],
        temperal_downsample=config["temperal_downsample"],
        dropout=config["dropout"],
        latents_mean=config["latents_mean"],
        latents_std=config["latents_std"],
        input_channels=input_channels,
        spatial_chunk_size=spatial_chunk_size,
    )

    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)
    state_dict = convert_3d_state_dict_to_2d(state_dict)

    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded image-only QwenImage 2D VAE: {info}")

    vae.to(device)
    return vae


__all__ = [
    "SCALE_FACTOR",
    "QwenImageResidualBlock2D",
    "QwenImageAttentionBlock2D",
    "QwenImageResample2D",
    "QwenImageMidBlock2D",
    "QwenImageEncoder2D",
    "QwenImageUpBlock2D",
    "QwenImageDecoder2D",
    "AutoencoderKLQwenImage2D",
    "convert_3d_state_dict_to_2d",
    "load_vae",
]
