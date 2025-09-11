from typing import Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Conv2d
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from library.utils import load_safetensors, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


VAE_SCALE_FACTOR = 32  # 32x spatial compression


def swish(x: Tensor) -> Tensor:
    """Swish activation function: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """Self-attention block using scaled dot-product attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b (h w) c").contiguous()

        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    Residual block with two convolutions, group normalization, and swish activation.
    Includes skip connection with optional channel dimension matching.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Skip connection projection for channel dimension mismatch
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        # First convolution block
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        # Second convolution block
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # Apply skip connection with optional projection
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """
    Spatial downsampling block that reduces resolution by 2x using convolution followed by
    pixel rearrangement. Includes skip connection with grouped averaging.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (must be divisible by 4).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4  # 2x2 spatial reduction factor
        assert out_channels % factor == 0

        self.conv = Conv2d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor) -> Tensor:
        # Apply convolution and rearrange pixels for 2x downsampling
        h = self.conv(x)
        h = rearrange(h, "b c (h r1) (w r2) -> b (r1 r2 c) h w", r1=2, r2=2)

        # Create skip connection with pixel rearrangement
        shortcut = rearrange(x, "b c (h r1) (w r2) -> b (r1 r2 c) h w", r1=2, r2=2)
        B, C, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, H, W).mean(dim=2)

        return h + shortcut


class Upsample(nn.Module):
    """
    Spatial upsampling block that increases resolution by 2x using convolution followed by
    pixel rearrangement. Includes skip connection with channel repetition.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        factor = 4  # 2x2 spatial expansion factor
        self.conv = Conv2d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor) -> Tensor:
        # Apply convolution and rearrange pixels for 2x upsampling
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 c) h w -> b c (h r1) (w r2)", r1=2, r2=2)

        # Create skip connection with channel repetition
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 c) h w -> b c (h r1) (w r2)", r1=2, r2=2)

        return h + shortcut


class Encoder(nn.Module):
    """
    VAE encoder that progressively downsamples input images to a latent representation.
    Uses residual blocks, attention, and spatial downsampling.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g., 3 for RGB).
    z_channels : int
        Number of latent channels in the output.
    block_out_channels : Tuple[int, ...]
        Output channels for each downsampling block.
    num_res_blocks : int
        Number of residual blocks per downsampling stage.
    ffactor_spatial : int
        Total spatial downsampling factor (e.g., 32 for 32x compression).
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.conv_in = Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]

        # Build downsampling blocks
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch

            # Add residual blocks for this level
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block

            # Add spatial downsampling if needed
            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            if add_spatial_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1]
                down.downsample = Downsample(block_in, block_out)
                block_in = block_out

            self.down.append(down)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # Initial convolution
        h = self.conv_in(x)

        # Progressive downsampling through blocks
        for i_level in range(len(self.block_out_channels)):
            # Apply residual blocks at this level
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            # Apply spatial downsampling if available
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # Middle processing with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Final output layers with skip connection
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) h w -> b c r h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """
    VAE decoder that progressively upsamples latent representations back to images.
    Uses residual blocks, attention, and spatial upsampling.

    Parameters
    ----------
    z_channels : int
        Number of latent channels in the input.
    out_channels : int
        Number of output image channels (e.g., 3 for RGB).
    block_out_channels : Tuple[int, ...]
        Output channels for each upsampling block.
    num_res_blocks : int
        Number of residual blocks per upsampling stage.
    ffactor_spatial : int
        Total spatial upsampling factor (e.g., 32 for 32x expansion).
    """

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Build upsampling blocks
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch

            # Add residual blocks for this level (extra block for decoder)
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block

            # Add spatial upsampling if needed
            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            if add_spatial_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1]
                up.upsample = Upsample(block_in, block_out)
                block_in = block_out

            self.up.append(up)

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # Initial processing with skip connection
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # Middle processing with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Progressive upsampling through blocks
        for i_level in range(len(self.block_out_channels)):
            # Apply residual blocks at this level
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            # Apply spatial upsampling if available
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # Final output layers
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class HunyuanVAE2D(nn.Module):
    """
    VAE model for Hunyuan Image-2.1 with spatial tiling support.

    This VAE uses a fixed architecture optimized for the Hunyuan Image-2.1 model,
    with 32x spatial compression and optional memory-efficient tiling for large images.
    """

    def __init__(self):
        super().__init__()

        # Fixed configuration for Hunyuan Image-2.1
        block_out_channels = (128, 256, 512, 512, 1024, 1024)
        in_channels = 3  # RGB input
        out_channels = 3  # RGB output
        latent_channels = 64
        layers_per_block = 2
        ffactor_spatial = 32  # 32x spatial compression
        sample_size = 384  # Minimum sample size for tiling
        scaling_factor = 0.75289  # Latent scaling factor

        self.ffactor_spatial = ffactor_spatial
        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
        )

        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
        )

        # Spatial tiling configuration for memory efficiency
        self.use_spatial_tiling = False
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_overlap_factor = 0.25  # 25% overlap between tiles

    @property
    def dtype(self):
        """Get the data type of the model parameters."""
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.encoder.parameters()).device

    def enable_spatial_tiling(self, use_tiling: bool = True):
        """Enable or disable spatial tiling."""
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        """Disable spatial tiling."""
        self.use_spatial_tiling = False

    def enable_tiling(self, use_tiling: bool = True):
        """Enable or disable spatial tiling (alias for enable_spatial_tiling)."""
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        """Disable spatial tiling (alias for disable_spatial_tiling)."""
        self.disable_spatial_tiling()

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        Blend two tensors horizontally with smooth transition.

        Parameters
        ----------
        a : torch.Tensor
            Left tensor.
        b : torch.Tensor
            Right tensor.
        blend_extent : int
            Number of columns to blend.
        """
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """
        Blend two tensors vertically with smooth transition.

        Parameters
        ----------
        a : torch.Tensor
            Top tensor.
        b : torch.Tensor
            Bottom tensor.
        blend_extent : int
            Number of rows to blend.
        """
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode large images using spatial tiling to reduce memory usage.
        Tiles are processed independently and blended at boundaries.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, T, H, W).
        """
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode large latents using spatial tiling to reduce memory usage.
        Tiles are processed independently and blended at boundaries.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (B, C, H, W).
        """
        B, C, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        return dec

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        """
        Encode input images to latent representation.
        Uses spatial tiling for large images if enabled.

        Parameters
        ----------
        x : Tensor
            Input image tensor of shape (B, C, H, W) or (B, C, T, H, W).

        Returns
        -------
        DiagonalGaussianDistribution
            Latent distribution with mean and logvar.
        """
        # Handle 5D input (B, C, T, H, W) by removing time dimension
        original_ndim = x.ndim
        if original_ndim == 5:
            x = x.squeeze(2)

        # Use tiling for large images to reduce memory usage
        if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            h = self.spatial_tiled_encode(x)
        else:
            h = self.encoder(x)

        # Restore time dimension if input was 5D
        if original_ndim == 5:
            h = h.unsqueeze(2)

        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z: Tensor):
        """
        Decode latent representation back to images.
        Uses spatial tiling for large latents if enabled.

        Parameters
        ----------
        z : Tensor
            Latent tensor of shape (B, C, H, W) or (B, C, T, H, W).

        Returns
        -------
        Tensor
            Decoded image tensor.
        """
        # Handle 5D input (B, C, T, H, W) by removing time dimension
        original_ndim = z.ndim
        if original_ndim == 5:
            z = z.squeeze(2)

        # Use tiling for large latents to reduce memory usage
        if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            decoded = self.spatial_tiled_decode(z)
        else:
            decoded = self.decoder(z)

        # Restore time dimension if input was 5D
        if original_ndim == 5:
            decoded = decoded.unsqueeze(2)

        return decoded


def load_vae(vae_path: str, device: torch.device, disable_mmap: bool = False) -> HunyuanVAE2D:
    logger.info("Initializing VAE")
    vae = HunyuanVAE2D()

    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)
    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")

    vae.to(device)
    return vae
