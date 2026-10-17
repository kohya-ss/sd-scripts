# FLUX.2 VAE (AutoEncoder with 32 latent channels, 2x2 patchify and BatchNorm normalization)
# Original code: https://github.com/black-forest-labs/flux (Apache-2.0 License)
# Ported from Musubi Tuner's flux2_models.py (https://github.com/kohya-ss/musubi-tuner)
#
# The state dict keys follow the original (BFL / ComfyUI) format. Weights in the diffusers format (AutoencoderKLFlux2)
# are converted on load. The decoder width is detected from the weights: the standard FLUX.2 VAE uses ch=128 for both
# the encoder and the decoder, while the VAE bundled with MOSS-Image uses the "FLUX.2 small decoder" (ch=96).

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from torch import Tensor, nn

from library.safetensors_utils import load_split_weights
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

LATENT_CHANNELS = 32  # channels of the raw VAE latents
LATENT_SCALE_FACTOR = 8  # pixel / raw latent
PATCH_SIZE = 2  # raw latent / DiT input (2x2 patchify -> 128 channels)
PATCHIFIED_LATENT_CHANNELS = LATENT_CHANNELS * PATCH_SIZE * PATCH_SIZE
BN_EPS = 1e-4


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32
    decoder_ch: Optional[int] = None  # None: same as ch. 96 for the FLUX.2 small decoder


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, 1, c, h * w).transpose(2, 3).contiguous()  # b 1 (h w) c
        k = k.reshape(b, 1, c, h * w).transpose(2, 3).contiguous()
        v = v.reshape(b, 1, c, h * w).transpose(2, 3).contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return h_.transpose(2, 3).reshape(b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
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
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


# region latent helpers (usable without the VAE instance, e.g. in the trainer with cached BN statistics)


def patchify_latents(z: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, C * 4, H / 2, W / 2] (2x2 patchify, channel order (c, pi, pj))"""
    b, c, h, w = z.shape
    return z.reshape(b, c, h // PATCH_SIZE, PATCH_SIZE, w // PATCH_SIZE, PATCH_SIZE).permute(0, 1, 3, 5, 2, 4).reshape(
        b, c * PATCH_SIZE * PATCH_SIZE, h // PATCH_SIZE, w // PATCH_SIZE
    )


def unpatchify_latents(z: Tensor) -> Tensor:
    """[B, C * 4, H / 2, W / 2] -> [B, C, H, W]"""
    b, c4, h, w = z.shape
    c = c4 // (PATCH_SIZE * PATCH_SIZE)
    return z.reshape(b, c, PATCH_SIZE, PATCH_SIZE, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * PATCH_SIZE, w * PATCH_SIZE)


def normalize_latents(z: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = BN_EPS) -> Tensor:
    """BatchNorm (eval) normalization of patchified latents with the given statistics."""
    s = torch.sqrt(running_var.to(z) + eps).view(1, -1, 1, 1)
    m = running_mean.to(z).view(1, -1, 1, 1)
    return (z - m) / s


def denormalize_latents(z: Tensor, running_mean: Tensor, running_var: Tensor, eps: float = BN_EPS) -> Tensor:
    """Inverse of `normalize_latents`."""
    s = torch.sqrt(running_var.to(z) + eps).view(1, -1, 1, 1)
    m = running_mean.to(z).view(1, -1, 1, 1)
    return z * s + m


# endregion


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.decoder_ch if params.decoder_ch is not None else params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = BN_EPS
        self.bn_momentum = 0.1
        self.ps = [PATCH_SIZE, PATCH_SIZE]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    @property
    def latent_channels(self) -> int:
        return self.params.z_channels

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def normalize(self, z: Tensor) -> Tensor:
        """Normalize patchified latents [B, 128, H/16, W/16] with the BatchNorm running statistics."""
        return normalize_latents(z, self.bn.running_mean, self.bn.running_var, self.bn_eps)

    def inv_normalize(self, z: Tensor) -> Tensor:
        return denormalize_latents(z, self.bn.running_mean, self.bn.running_var, self.bn_eps)

    def encode(self, x: Tensor) -> Tensor:
        """Encode pixels [B, 3, H, W] (range -1..1) to raw latents [B, 32, H/8, W/8] (the mean of the distribution).

        Patchify and normalization are not applied: use `patchify_and_normalize` (or `encode_for_dit`) for the DiT input.
        """
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        return mean

    def patchify_and_normalize(self, z: Tensor) -> Tensor:
        """Raw latents [B, 32, H/8, W/8] -> DiT input [B, 128, H/16, W/16]"""
        return self.normalize(patchify_latents(z))

    def denormalize_and_unpatchify(self, z: Tensor) -> Tensor:
        """DiT output [B, 128, H/16, W/16] -> raw latents [B, 32, H/8, W/8]"""
        return unpatchify_latents(self.inv_normalize(z))

    def encode_for_dit(self, x: Tensor) -> Tensor:
        return self.patchify_and_normalize(self.encode(x))

    def decode(self, z: Tensor) -> Tensor:
        """Decode raw latents [B, 32, H/8, W/8] to pixels [B, 3, H, W]"""
        return self.decoder(z)

    def decode_from_dit(self, z: Tensor) -> Tensor:
        return self.decode(self.denormalize_and_unpatchify(z))


# region loading


def _convert_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a diffusers AutoencoderKLFlux2 state dict to the original (BFL) key names."""
    new_sd = {}
    num_levels = len({k.split(".")[2] for k in sd if k.startswith("decoder.up_blocks.")})

    def convert_resnet(key: str) -> str:
        return key.replace("conv_shortcut.", "nin_shortcut.")

    for key, value in sd.items():
        new_key = key
        if key.startswith("quant_conv."):
            new_key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            new_key = "decoder." + key
        elif ".mid_block." in key:
            prefix, rest = key.split(".mid_block.", 1)
            if rest.startswith("resnets."):
                _, idx, rest2 = rest.split(".", 2)
                new_key = f"{prefix}.mid.block_{int(idx) + 1}.{convert_resnet(rest2)}"
            elif rest.startswith("attentions."):
                _, idx, rest2 = rest.split(".", 2)
                mapping = {
                    "group_norm": "norm",
                    "to_q": "q",
                    "to_k": "k",
                    "to_v": "v",
                    "to_out.0": "proj_out",
                }
                for src, dst in mapping.items():
                    if rest2.startswith(src + "."):
                        rest2 = dst + rest2[len(src) :]
                        break
                new_key = f"{prefix}.mid.attn_1.{rest2}"
                if rest2.endswith(".weight") and value.ndim == 2:
                    value = value[:, :, None, None]  # Linear -> 1x1 Conv2d
        elif key.startswith("encoder.down_blocks."):
            _, _, idx, rest = key.split(".", 3)
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("downsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.downsample.{rest2}"
        elif key.startswith("decoder.up_blocks."):
            _, _, idx, rest = key.split(".", 3)
            level = num_levels - 1 - int(idx)  # diffusers up_blocks.0 is the deepest level (BFL up.{n-1})
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("upsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.upsample.{rest2}"
        elif key.startswith("encoder.conv_norm_out.") or key.startswith("decoder.conv_norm_out."):
            new_key = key.replace(".conv_norm_out.", ".norm_out.")
        new_sd[new_key] = value
    return new_sd


def is_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> bool:
    return any(k.startswith("encoder.down_blocks.") for k in sd)


def load_flux2_vae(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False
) -> AutoEncoder:
    """Load the FLUX.2 VAE from a safetensors file (BFL / ComfyUI format or diffusers format)."""
    logger.info(f"Loading FLUX.2 VAE from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap)
    if dtype is not None:
        sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}  # keep bn.num_batches_tracked as is

    # remove prefixes used by some single-file checkpoints
    for prefix in ["vae.", "first_stage_model."]:
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}

    if is_diffusers_vae_state_dict(sd):
        logger.info("Converting diffusers format VAE state dict to the original format")
        sd = _convert_diffusers_vae_state_dict(sd)

    params = AutoEncoderParams()
    decoder_ch = sd["decoder.conv_out.weight"].shape[1]  # ch * ch_mult[0] = ch
    if decoder_ch != params.ch:
        logger.info(f"Decoder width {decoder_ch} differs from the encoder width {params.ch} (FLUX.2 small decoder)")
        params.decoder_ch = decoder_ch

    with init_empty_weights():
        ae = AutoEncoder(params)
    if dtype is not None:
        ae.to(dtype)

    info = ae.load_state_dict(sd, strict=True, assign=True)
    ae.to(device)  # buffers created by init_empty_weights (bn.num_batches_tracked) are on the CPU
    ae.bn.eval()
    logger.info(f"Loaded FLUX.2 VAE: {info}")
    return ae


# endregion
