import os
from typing import Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.utils

from diffusers import AutoencoderKL, LMSDiscreteScheduler
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

from .hunyuan_models import MT5Embedder, HunYuanDiT, BertModel, DiT_g_2


def get_input_ids(caption, tokenizer, tokenizer_max_length=225):
    tokens = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    masks = tokens["attention_mask"]

    if tokenizer_max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        masks = masks.squeeze(0)
        iids_list = []
        mask_list = []
        for i in range(
            1,
            tokenizer_max_length - tokenizer.model_max_length + 2,
            tokenizer.model_max_length - 2,
        ):
            ids_chunk = (
                input_ids[0].unsqueeze(0),  # BOS
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),
            )  # PAD or EOS
            ids_chunk = torch.cat(ids_chunk)
            mask_chunk = (
                masks[0].unsqueeze(0),
                masks[i : i + tokenizer.model_max_length - 2],
                masks[-1].unsqueeze(0),
            )
            mask_chunk = torch.cat(mask_chunk)

            # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
            # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
            if (
                ids_chunk[-2] != tokenizer.eos_token_id
                and ids_chunk[-2] != tokenizer.pad_token_id
            ):
                ids_chunk[-1] = tokenizer.eos_token_id
            # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
            if ids_chunk[1] == tokenizer.pad_token_id:
                ids_chunk[1] = tokenizer.eos_token_id

            iids_list.append(ids_chunk)
            mask_list.append(mask_chunk)

        input_ids = torch.stack(iids_list)  # 3,77
        masks = torch.stack(mask_list)  # 3,77
    return input_ids, masks


def get_hidden_states(
    input_ids,
    masks,
    tokenizer,
    text_encoder: BertModel,
    max_token_length=225,
    weight_dtype=None,
):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size(0)
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77
    masks = masks.reshape((-1, tokenizer.model_max_length))

    encoder_hidden_states = text_encoder(input_ids, attention_mask=masks)[0]

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape(
        (b_size, -1, encoder_hidden_states.shape[-1])
    )
    masks = masks.reshape((b_size, -1))

    if max_token_length is not None:
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        mask_list = [masks[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer.model_max_length):
            states_list.append(
                encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]
            )  # <BOS> の後から <EOS> の前まで
            mask_list.append(masks[:, i : i + tokenizer.model_max_length - 2])
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        mask_list.append(masks[:, -1].unsqueeze(1))

        masks = torch.cat(mask_list, dim=1)
        encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1])), masks


def get_cond(
    prompt: str,
    mt5_embedder: MT5Embedder,
    clip_tokenizer: BertTokenizer,
    clip_encoder: BertModel,
    max_length_clip: int = 75 * 3 + 2,
    dtype=None,
    device="cuda",
):
    '''
    Get CLIP and mT5 embeddings for HunYuan DiT
    Note that this function support "CLIP Concat" trick.
    with max_length_clip = 152/227 or higher.
    '''
    prompt = prompt.strip()
    clip_input_ids, mask = get_input_ids(prompt, clip_tokenizer, max_length_clip)
    clip_hidden_states, clip_mask = get_hidden_states(
        clip_input_ids.unsqueeze(0).to(device),
        mask.to(device),
        clip_tokenizer,
        clip_encoder,
        max_token_length=max_length_clip,
    )

    mt5_hidden_states, mt5_mask = mt5_embedder.get_text_embeddings(prompt)

    return (
        clip_hidden_states.to(dtype),
        clip_mask.long().to(device),
        mt5_hidden_states.to(dtype),
        mt5_mask.long().to(device),
    )


def load_scheduler_sigmas():
    scheduler: LMSDiscreteScheduler = LMSDiscreteScheduler.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        subfolder="scheduler",
    )
    return scheduler.alphas_cumprod, scheduler.sigmas


def load_model(model_path: str, dtype=torch.float16, device="cuda"):
    denoiser: HunYuanDiT
    denoiser, patch_size, head_dim = DiT_g_2(input_size=(128, 128))
    sd = torch.load(os.path.join(model_path, "denoiser/pytorch_model_module.pt"))
    denoiser.load_state_dict(sd)
    denoiser.to(device).to(dtype)

    clip_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "clip"))
    clip_tokenizer.eos_token_id = 2
    clip_encoder = (
        BertModel.from_pretrained(os.path.join(model_path, "clip")).to(device).to(dtype)
    )

    mt5_embedder = (
        MT5Embedder(os.path.join(model_path, "mt5"), torch_dtype=dtype, max_length=256)
        .to(device)
        .to(dtype)
    )
    mt5_embedder.device = device

    vae = (
        AutoencoderKL.from_pretrained(os.path.join(model_path, "vae"))
        .to(device)
        .to(dtype)
    )
    vae.requires_grad_(False)
    return (
        denoiser,
        patch_size,
        head_dim,
        clip_tokenizer,
        clip_encoder,
        mt5_embedder,
        vae,
    )


def _to_tuple(x):
    if isinstance(x, int):
        return x, x
    else:
        return x


def get_fill_resize_and_crop(src, tgt):
    th, tw = _to_tuple(tgt)
    h, w = _to_tuple(src)

    tr = th / tw  # base resolution
    r = h / w  # target resolution

    # resize
    if r > tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(
            round(tw / w * h)
        )  # resize the target resolution down based on the base resolution

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def get_meshgrid(start, *args):
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start)
        start = (0, 0)
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = (stop[0] - start[0], stop[1] - start[1])
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = _to_tuple(args[1])
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # [2, W, H]
    return grid


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, start, *args, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = get_meshgrid(start, *args)  # [2, H, w]
    # grid_h = np.arange(grid_size, dtype=np.float32)
    # grid_w = np.arange(grid_size, dtype=np.float32)
    # grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # grid = np.stack(grid, axis=0)   # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
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
    pos: a list of positions to be encoded: size (W,H)
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


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/facebookresearch/llama/blob/main/llama/model.py#L443


def get_2d_rotary_pos_embed(embed_dim, start, *args, use_real=True):
    """
    This is a 2d version of precompute_freqs_cis, which is a RoPE for image tokens with 2d structure.

    Parameters
    ----------
    embed_dim: int
        embedding dimension size
    start: int or tuple of int
        If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop, step is 1;
        If len(args) == 2, start is start, args[0] is stop, args[1] is num.
    use_real: bool
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns
    -------
    pos_embed: torch.Tensor
        [HW, D/2]
    """
    grid = get_meshgrid(start, *args)  # [2, H, w]
    grid = grid.reshape(
        [2, 1, *grid.shape[1:]]
    )  # Returns a sampling matrix with the same resolution as the target resolution
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D/2)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D/2)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (np.ndarray, int): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


def calc_sizes(rope_img, patch_size, th, tw):
    if rope_img == "extend":
        # Expansion mode
        sub_args = [(th, tw)]
    elif rope_img.startswith("base"):
        # Based on the specified dimensions, other dimensions are obtained through interpolation.
        base_size = int(rope_img[4:]) // 8 // patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
    else:
        raise ValueError(f"Unknown rope_img: {rope_img}")
    return sub_args


def init_image_posemb(
    rope_img,
    resolutions,
    patch_size,
    hidden_size,
    num_heads,
    log_fn,
    rope_real=True,
):
    freqs_cis_img = {}
    for reso in resolutions:
        th, tw = reso.height // 8 // patch_size, reso.width // 8 // patch_size
        sub_args = calc_sizes(rope_img, patch_size, th, tw)
        freqs_cis_img[str(reso)] = get_2d_rotary_pos_embed(
            hidden_size // num_heads, *sub_args, use_real=rope_real
        )
        log_fn(
            f"    Using image RoPE ({rope_img}) ({'real' if rope_real else 'complex'}): {sub_args} | ({reso}) "
            f"{freqs_cis_img[str(reso)][0].shape if rope_real else freqs_cis_img[str(reso)].shape}"
        )
    return freqs_cis_img


def calc_rope(height, width, patch_size=2, head_size=64):
    th = height // 8 // patch_size
    tw = width // 8 // patch_size
    base_size = 512 // 8 // patch_size
    start, stop = get_fill_resize_and_crop((th, tw), base_size)
    sub_args = [start, stop, (th, tw)]
    rope = get_2d_rotary_pos_embed(head_size, *sub_args)
    return rope


if __name__ == "__main__":
    clip_tokenizer = AutoTokenizer.from_pretrained("./model/clip")
    clip_tokenizer.eos_token_id = 2
    clip_encoder = BertModel.from_pretrained("./model/clip").half().cuda()
    print(clip_tokenizer.eos_token_id, clip_tokenizer.eos_token)

    mt5_embedder = MT5Embedder(
        "./model/mt5", torch_dtype=torch.float16, max_length=256
    ).cuda()
    mt5_embedder.device = "cuda"

    clip_h, clip_m, mt5_h, mt5_m = get_cond("""anime style, illustration, masterpiece,
1girl,

ciloranko, maccha (mochancc), lobelia (saclia), welchino, yanyo (ogino atsuki),

solo, loli, purple eyes, hair ornament, dragon wings, wings, indoors, dragon girl, 
dragon tail, pointy ears, long hair, two side up, earrings, medium breasts, 
cleavage, swimsuit, sitting, breasts, jewelry, barefoot, bikini, tail, horns, 
looking at viewer, smile, dragon horns, collarbone, feet, navel, purple hair, full body, bare legs,

masterpiece, newest, absurdres, sensitive
""",
        mt5_embedder,
        clip_tokenizer,
        clip_encoder,
        75*3+2
    )
    print(clip_h.dtype, clip_m.dtype, mt5_h.dtype, mt5_m.dtype)
    print(clip_h.shape, clip_m.shape, mt5_h.shape, mt5_m.shape)
    print(mt5_m)
    print(clip_m)
