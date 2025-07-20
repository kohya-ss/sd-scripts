# copy from the official repo: https://github.com/lodestone-rock/flow/blob/master/src/models/chroma/model.py
# and modified
# licensed under Apache License 2.0

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .flux_models import attention, rope, apply_rope, EmbedND, timestep_embedding, MLPEmbedder, RMSNorm, QKNorm, SelfAttention, Flux
from . import custom_offloading_utils


def distribute_modulations(tensor: torch.Tensor, depth_single_blocks, depth_double_blocks):
    """
    Distributes slices of the tensor into the block_dict as ModulationOut objects.

    Args:
        tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
    """
    batch_size, vectors, dim = tensor.shape

    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(depth_single_blocks):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(depth_double_blocks):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None
    # 6.2b version
    # block_dict["lite_double_blocks.4.img_mod.lin"] = None
    # block_dict["lite_double_blocks.4.txt_mod.lin"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            # Single block: 1 ModulationOut
            block_dict[key] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "txt_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "final_layer" in key:
            # Final layer: 1 ModulationOut
            block_dict[key] = [
                tensor[:, idx : idx + 1, :],
                tensor[:, idx + 1 : idx + 2, :],
            ]
            idx += 2  # Advance by 3 vectors

    return block_dict


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def enable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.disable_gradient_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift


def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.gradient_checkpointing = False

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        return _modulation_gate_fn(x, gate, gate_params)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        # replaced with compiled fn
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_modulated = self.modulation_shift_scale_fn(img_modulated, img_mod1.scale, img_mod1.shift)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        # replaced with compiled fn
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_modulated = self.modulation_shift_scale_fn(txt_modulated, txt_mod1.scale, txt_mod1.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # replaced with compiled fn
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        img = self.modulation_gate_fn(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = self.modulation_gate_fn(
            img,
            img_mod2.gate,
            self.img_mlp(self.modulation_shift_scale_fn(self.img_norm2(img), img_mod2.scale, img_mod2.shift)),
        )

        # calculate the txt bloks
        # replaced with compiled fn
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = self.modulation_gate_fn(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = self.modulation_gate_fn(
            txt,
            txt_mod2.gate,
            self.txt_mlp(self.modulation_shift_scale_fn(self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift)),
        )

        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training and self.gradient_checkpointing:
            return ckpt.checkpoint(self._forward, img, txt, pe, distill_vec, mask, use_reentrant=False)
        else:
            return self._forward(img, txt, pe, distill_vec, mask)


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

        self.gradient_checkpointing = False

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        return _modulation_gate_fn(x, gate, gate_params)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor) -> Tensor:
        mod = distill_vec
        # replaced with compiled fn
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = self.modulation_shift_scale_fn(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # replaced with compiled fn
        # return x + mod.gate * output
        return self.modulation_gate_fn(x, mod.gate, output)

    def forward(self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor) -> Tensor:
        if self.training and self.gradient_checkpointing:
            return ckpt.checkpoint(self._forward, x, pe, distill_vec, mask, use_reentrant=False)
        else:
            return self._forward(x, pe, distill_vec, mask)


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        return _modulation_shift_scale_fn(x, scale, shift)

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        # replaced with compiled fn
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.modulation_shift_scale_fn(self.norm_final(x), scale[:, None, :], shift[:, None, :])
        x = self.linear(x)
        return x


@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    _use_compiled: bool


chroma_params = ChromaParams(
    in_channels=64,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
    approximator_in_dim=64,
    approximator_depth=5,
    approximator_hidden_size=5120,
    _use_compiled=False,
)


def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask


class Chroma(Flux):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: ChromaParams):
        nn.Module.__init__(self)
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # TODO: need proper mapping for this approximator output!
        # currently the mapping is hardcoded in distribute_modulations function
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
        )

        # TODO: move this hardcoded value to config
        # single layer has 3 modulation vectors
        # double layer has 6 modulation vectors for each expert
        # final layer has 2 modulation vectors
        self.mod_index_length = 3 * params.depth_single_blocks + 2 * 6 * params.depth + 2
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth
        # self.mod_index = torch.tensor(list(range(self.mod_index_length)), device=0)
        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length)), device="cpu"),
            persistent=False,
        )
        self.approximator_in_dim = params.approximator_in_dim

        self.blocks_to_swap = None
        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

        # Initialize properties required by Flux parent class
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        block_controlnet_hidden_states=None,
        block_controlnet_single_hidden_states=None,
        guidance: Tensor | None = None,
        txt_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # TODO:
        # need to fix grad accumulation issue here for now it's in no grad mode
        # besides, i don't want to wash out the PFP that's trained on this model weights anyway
        # the fan out operation here is deleting the backward graph
        # alternatively doing forward pass for every block manually is doable but slow
        # custom backward probably be better
        with torch.no_grad():
            distill_timestep = timestep_embedding(timesteps, self.approximator_in_dim // 4)
            # TODO: need to add toggle to omit this from schnell but that's not a priority
            distil_guidance = timestep_embedding(guidance, self.approximator_in_dim // 4)
            # get all modulation index
            modulation_index = timestep_embedding(self.mod_index, self.approximator_in_dim // 2)
            # we need to broadcast the modulation index here so each batch has all of the index
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            # and we need to broadcast timestep and guidance along too
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, self.mod_index_length, 1)
            )
            # then and only then we could concatenate it together
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))
        mod_vectors_dict = distribute_modulations(mod_vectors, self.depth_single_blocks, self.depth_double_blocks)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # compute mask
        # assume max seq length from the batched input

        max_len = txt.shape[1]

        # mask
        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(txt_attention_mask, max_len, 1)
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_attention_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = txt_img_mask[None, None, ...].repeat(txt.shape[0], self.num_heads, 1, 1).int().bool()
            # txt_mask_w_padding[txt_mask_w_padding==False] = True

        if not self.blocks_to_swap:
            for i, block in enumerate(self.double_blocks):
                # the guidance replaced by FFN output
                img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
                txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
                double_mod = [img_mod, txt_mod]

                img, txt = block(img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask)
        else:
            for i, block in enumerate(self.double_blocks):
                self.offloader_double.wait_for_block(i)

                # the guidance replaced by FFN output
                img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
                txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
                double_mod = [img_mod, txt_mod]

                img, txt = block(img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask)

                self.offloader_double.submit_move_blocks(self.double_blocks, i)

        img = torch.cat((txt, img), 1)
        if not self.blocks_to_swap:
            for i, block in enumerate(self.single_blocks):
                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
                img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)
        else:
            for i, block in enumerate(self.single_blocks):
                self.offloader_single.wait_for_block(i)

                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
                img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)

                self.offloader_single.submit_move_blocks(self.single_blocks, i)
        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, distill_vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img
