from collections.abc import Mapping
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import math
import argparse
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.types import Number
from typing import List, Optional, Union, Protocol
from .utils import setup_logging

try:
    import pywt
except:
    pass


setup_logging()
import logging

logger = logging.getLogger(__name__)


def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.info(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(
    loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, gamma: Number, v_prediction=False
):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def scale_v_prediction_loss_like_noise_prediction(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler):
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.info(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(
    loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, v_pred_like_loss: torch.Tensor
):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.info(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss


def apply_debiased_estimation(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, v_prediction=False):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


# TODO train_utilと分散しているのでどちらかに寄せる


def add_custom_train_arguments(parser: argparse.ArgumentParser, support_weighted_captions: bool = True):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper. / 低いタイムステップでの高いlossに対して重みを減らすためのgamma値、低いほど効果が強く、論文では5が推奨",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss / v-prediction lossをnoise prediction lossと同じようにスケーリングする",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value / v-prediction lossをこの値をかけたものをlossに加算する",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss / debiased estimation loss",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="Activate wavelet loss. Default: False")
    parser.add_argument("--wavelet_loss_primary", action="store_true", help="Use wavelet loss as the primary loss")
    parser.add_argument("--wavelet_loss_alpha", type=float, default=1.0, help="Wavelet loss alpha. Default: 1.0")
    parser.add_argument("--wavelet_loss_type", help="Wavelet loss type l1, l2, huber, smooth_l1. Default to --loss_type value.")
    parser.add_argument("--wavelet_loss_transform", default="swt", help="Wavelet transform type of DWT or SWT. Default: swt")
    parser.add_argument("--wavelet_loss_wavelet", default="sym7", help="Wavelet. Default: sym7")
    parser.add_argument(
        "--wavelet_loss_level",
        type=int,
        default=1,
        help="Wavelet loss level 1 (main) or 2 (details). Higher levels are available for DWT for higher resolution training. Default: 1",
    )
    parser.add_argument(
        "--wavelet_loss_rectified_flow", type=bool, default=True, help="Use rectified flow to estimate clean latents before wavelet loss"
    )
    import ast
    import json

    def parse_wavelet_weights(weights_str):
        if weights_str is None:
            return None

        # Try parsing as a dictionary (for formats like "{'ll1':0.1,'lh1':0.01}")
        if weights_str.strip().startswith("{"):
            try:
                return ast.literal_eval(weights_str)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(weights_str.replace("'", '"'))
                except json.JSONDecodeError:
                    pass

        # Parse format like "ll1=0.1,lh1=0.01,hl1=0.01,hh1=0.05"
        result = {}
        for pair in weights_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key.strip()] = float(value.strip())

        return result

    parser.add_argument(
        "--wavelet_loss_band_level_weights",
        type=parse_wavelet_weights,
        default=None,
        help="Wavelet loss band level weights. ll1=0.1,lh1=0.01,hl1=0.01,hh1=0.05. Default: None",
    )
    parser.add_argument(
        "--wavelet_loss_band_weights",
        type=parse_wavelet_weights,
        default=None,
        help="Wavelet loss band weights. ll=0.1,lh=0.01,hl=0.01,hh=0.05. Default: None",
    )
    parser.add_argument(
        "--wavelet_loss_quaternion_component_weights",
        type=parse_wavelet_weights,
        default=None,
        help="Quaternion Wavelet loss component weights r=1.0 real i=0.7 x-Hilbert j=0.7 y-Hilbert k=0.5 xy-Hilbert",
    )
    parser.add_argument(
        "--wavelet_loss_ll_level_threshold",
        default=None,
        type=int,
        help="Wavelet loss which level to calculate the loss for the low frequency (ll). -1 means last n level. Default: None",
    )
    parser.add_argument(
        "--wavelet_loss_energy_loss_ratio",
        type=float,
        help="Ratio for energy loss ratio between pattern loss differences in wavelets. ",
    )
    parser.add_argument(
        "--wavelet_loss_energy_scale_factor",
        type=float,
        help="Scale for energy loss",
    )
    parser.add_argument(
        "--wavelet_loss_normalize_bands",
        default=None,
        action="store_true",
        help="Normalize wavelet bands before calculating the loss.",
    )
    parser.add_argument(
        "--wavelet_loss_metrics",
        action="store_true",
        help="Create and log wavelet metrics.",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
        )


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    tokenizer,
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = text_encoder(text_input_chunk)[0]
            else:
                enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = text_encoder(text_input)[0]
        else:
            enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    clip_skip=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return text_embeddings


# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4) -> torch.FloatTensor:
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale) -> torch.FloatTensor:
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)  # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss


class LossCallableMSE(Protocol):
    def __call__(
        self,
        input: Tensor,
        target: Tensor,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> Tensor: ...


class LossCallableReduction(Protocol):
    def __call__(self, input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: ...


LossCallable = LossCallableReduction | LossCallableMSE


class WaveletTransform:
    """Base class for wavelet transforms."""

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters."""
        assert pywt.Wavelet is not None, "PyWavelets module not available. Please install `pip install PyWavelets`"

        # Create filters from wavelet
        wav = pywt.Wavelet(wavelet)
        self.dec_lo = torch.tensor(wav.dec_lo).to(device)
        self.dec_hi = torch.tensor(wav.dec_hi).to(device)

    def decompose(self, x: Tensor) -> dict[str, list[Tensor]]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("WaveletTransform subclasses must implement decompose method")


class DiscreteWaveletTransform(WaveletTransform):
    """Discrete Wavelet Transform (DWT) implementation."""

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """
        Perform multi-level DWT decomposition.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing decomposition coefficients
        """
        bands: dict[str, list[Tensor]] = {
            "ll": [],
            "lh": [],
            "hl": [],
            "hh": [],
        }

        # Start low frequency with input
        ll = x

        for _ in range(level):
            ll, lh, hl, hh = self._dwt_single_level(ll)

            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
            bands["ll"].append(ll)

        return bands

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level DWT decomposition."""
        batch, channels, height, width = x.shape
        x = x.view(batch * channels, 1, height, width)

        # Calculate proper padding for the filter size
        filter_size = self.dec_lo.size(0)
        pad_size = filter_size // 2

        # Pad for proper convolution
        try:
            x_pad = F.pad(x, (pad_size,) * 4, mode="reflect")
        except RuntimeError:
            # Fallback for very small tensors
            x_pad = F.pad(x, (pad_size,) * 4, mode="constant")

        # Apply filter to rows
        lo = F.conv2d(x_pad, self.dec_lo.view(1, 1, -1, 1), stride=(2, 1))
        hi = F.conv2d(x_pad, self.dec_hi.view(1, 1, -1, 1), stride=(2, 1))

        # Apply filter to columns
        ll = F.conv2d(lo, self.dec_lo.view(1, 1, 1, -1), stride=(1, 2))
        lh = F.conv2d(lo, self.dec_hi.view(1, 1, 1, -1), stride=(1, 2))
        hl = F.conv2d(hi, self.dec_lo.view(1, 1, 1, -1), stride=(1, 2))
        hh = F.conv2d(hi, self.dec_hi.view(1, 1, 1, -1), stride=(1, 2))

        # Reshape back to batch format
        ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(x.device)
        lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(x.device)
        hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(x.device)
        hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(x.device)

        return ll, lh, hl, hh


class StationaryWaveletTransform(WaveletTransform):
    """Stationary Wavelet Transform (SWT) implementation."""

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters."""
        super().__init__(wavelet, device)

        # Store original filters
        self.orig_dec_lo = self.dec_lo.clone()
        self.orig_dec_hi = self.dec_hi.clone()

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """Perform multi-level SWT decomposition."""
        bands = {
            "ll": [],  # or "aa" if you prefer PyWavelets nomenclature
            "lh": [],  # or "da"
            "hl": [],  # or "ad"
            "hh": [],  # or "dd"
        }

        # Start with input as low frequency
        ll = x

        for j in range(level):
            # Get upsampled filters for current level
            dec_lo, dec_hi = self._get_filters_for_level(j)

            # Decompose current approximation
            ll, lh, hl, hh = self._swt_single_level(ll, dec_lo, dec_hi)

            # Store results in bands
            bands["ll"].append(ll)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)

            # No need to update ll explicitly as it's already the next approximation

        return bands

    def _get_filters_for_level(self, level: int) -> tuple[Tensor, Tensor]:
        """Get upsampled filters for the specified level."""
        if level == 0:
            return self.orig_dec_lo, self.orig_dec_hi

        # Calculate number of zeros to insert
        zeros = 2**level - 1

        # Create upsampled filters
        upsampled_dec_lo = torch.zeros(len(self.orig_dec_lo) + (len(self.orig_dec_lo) - 1) * zeros, device=self.orig_dec_lo.device)
        upsampled_dec_hi = torch.zeros(len(self.orig_dec_hi) + (len(self.orig_dec_hi) - 1) * zeros, device=self.orig_dec_hi.device)

        # Insert original coefficients with zeros in between
        upsampled_dec_lo[:: zeros + 1] = self.orig_dec_lo
        upsampled_dec_hi[:: zeros + 1] = self.orig_dec_hi

        return upsampled_dec_lo, upsampled_dec_hi

    def _swt_single_level(self, x: Tensor, dec_lo: Tensor, dec_hi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level SWT decomposition with 1D convolutions."""
        batch, channels, height, width = x.shape

        # Prepare output tensors
        ll = torch.zeros((batch, channels, height, width), device=x.device)
        lh = torch.zeros((batch, channels, height, width), device=x.device)
        hl = torch.zeros((batch, channels, height, width), device=x.device)
        hh = torch.zeros((batch, channels, height, width), device=x.device)

        # Prepare 1D filter kernels
        dec_lo_1d = dec_lo.view(1, 1, -1)
        dec_hi_1d = dec_hi.view(1, 1, -1)
        pad_len = dec_lo.size(0) - 1

        for b in range(batch):
            for c in range(channels):
                # Extract single channel/batch and reshape for 1D convolution
                x_bc = x[b, c]  # Shape: [height, width]

                # Process rows with 1D convolution
                # Reshape to [width, 1, height] for treating each row as a batch
                x_rows = x_bc.transpose(0, 1).unsqueeze(1)  # Shape: [width, 1, height]

                # Pad for circular convolution
                x_rows_padded = F.pad(x_rows, (pad_len, 0), mode="circular")

                # Apply filters to rows
                x_lo_rows = F.conv1d(x_rows_padded, dec_lo_1d)  # [width, 1, height]
                x_hi_rows = F.conv1d(x_rows_padded, dec_hi_1d)  # [width, 1, height]

                # Reshape and transpose back
                x_lo_rows = x_lo_rows.squeeze(1).transpose(0, 1)  # [height, width]
                x_hi_rows = x_hi_rows.squeeze(1).transpose(0, 1)  # [height, width]

                # Process columns with 1D convolution
                # Reshape for column filtering (no transpose needed)
                x_lo_cols = x_lo_rows.unsqueeze(1)  # [height, 1, width]
                x_hi_cols = x_hi_rows.unsqueeze(1)  # [height, 1, width]

                # Pad for circular convolution
                x_lo_cols_padded = F.pad(x_lo_cols, (pad_len, 0), mode="circular")
                x_hi_cols_padded = F.pad(x_hi_cols, (pad_len, 0), mode="circular")

                # Apply filters to columns
                ll[b, c] = F.conv1d(x_lo_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                lh[b, c] = F.conv1d(x_lo_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]
                hl[b, c] = F.conv1d(x_hi_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                hh[b, c] = F.conv1d(x_hi_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]

        return ll, lh, hl, hh


class QuaternionWaveletTransform(WaveletTransform):
    """
    Quaternion Wavelet Transform implementation.
    Combines real DWT with three Hilbert transforms along x, y, and xy axes.
    """

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters and Hilbert transforms."""
        super().__init__(wavelet, device)

        # Register Hilbert transform filters
        self.register_hilbert_filters(device)

    def register_hilbert_filters(self, device):
        """Create and register Hilbert transform filters."""
        # Create x-axis Hilbert filter
        self.hilbert_x = self._create_hilbert_filter("x").to(device)

        # Create y-axis Hilbert filter
        self.hilbert_y = self._create_hilbert_filter("y").to(device)

        # Create xy (diagonal) Hilbert filter
        self.hilbert_xy = self._create_hilbert_filter("xy").to(device)

    def _create_hilbert_filter(self, direction):
        """Create a Hilbert transform filter for the specified direction."""
        if direction == "x":
            # Horizontal Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0106, -0.0329, -0.0308, 0.0000, 0.0308, 0.0329, 0.0106],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

        elif direction == "y":
            # Vertical Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0106, 0.0000],
                    [-0.0329, 0.0000],
                    [-0.0308, 0.0000],
                    [0.0000, 0.0000],
                    [0.0308, 0.0000],
                    [0.0329, 0.0000],
                    [0.0106, 0.0000],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

        else:  # 'xy' - diagonal
            # Diagonal Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0011, -0.0035, -0.0033, 0.0000, 0.0033, 0.0035, 0.0011],
                    [-0.0035, -0.0108, -0.0102, 0.0000, 0.0102, 0.0108, 0.0035],
                    [-0.0033, -0.0102, -0.0095, 0.0000, 0.0095, 0.0102, 0.0033],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0033, 0.0102, 0.0095, 0.0000, -0.0095, -0.0102, -0.0033],
                    [0.0035, 0.0108, 0.0102, 0.0000, -0.0102, -0.0108, -0.0035],
                    [0.0011, 0.0035, 0.0033, 0.0000, -0.0033, -0.0035, -0.0011],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

    def _apply_hilbert(self, x, direction):
        """Apply Hilbert transform in specified direction with correct padding."""
        batch, channels, height, width = x.shape

        x_flat = x.reshape(batch * channels, 1, height, width)

        # Get the appropriate filter
        if direction == "x":
            h_filter = self.hilbert_x
        elif direction == "y":
            h_filter = self.hilbert_y
        else:  # 'xy'
            h_filter = self.hilbert_xy

        # Calculate correct padding based on filter dimensions
        # For 'same' padding: pad = (filter_size - 1) / 2
        filter_h, filter_w = h_filter.shape[2:]
        pad_h = (filter_h - 1) // 2
        pad_w = (filter_w - 1) // 2

        # For even-sized filters, we need to adjust padding
        pad_h_left, pad_h_right = pad_h, pad_h
        pad_w_left, pad_w_right = pad_w, pad_w

        if filter_h % 2 == 0:  # Even height
            pad_h_right += 1
        if filter_w % 2 == 0:  # Even width
            pad_w_right += 1

        # Apply padding with possibly asymmetric padding
        x_pad = F.pad(x_flat, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

        # Apply convolution
        x_hilbert = F.conv2d(x_pad, h_filter)

        # Ensure output dimensions match input dimensions
        if x_hilbert.shape[2:] != (height, width):
            # Need to crop or pad to match original dimensions
            # For this case, center crop is appropriate
            if x_hilbert.shape[2] > height:
                # Crop height
                diff = x_hilbert.shape[2] - height
                start = diff // 2
                x_hilbert = x_hilbert[:, :, start : start + height, :]

            if x_hilbert.shape[3] > width:
                # Crop width
                diff = x_hilbert.shape[3] - width
                start = diff // 2
                x_hilbert = x_hilbert[:, :, :, start : start + width]

        # Reshape back to original format
        return x_hilbert.reshape(batch, channels, height, width)

    def decompose(self, x: Tensor, level=1) -> dict[str, dict[str, list[Tensor]]]:
        """
        Perform multi-level QWT decomposition.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing quaternion wavelet coefficients
            Format: {component: {band: [level1, level2, ...]}}
            where component ∈ {r, i, j, k} and band ∈ {ll, lh, hl, hh}
        """
        # Initialize result dictionary with quaternion components
        qwt_coeffs = {
            "r": {"ll": [], "lh": [], "hl": [], "hh": []},  # Real part
            "i": {"ll": [], "lh": [], "hl": [], "hh": []},  # Imaginary part (x-Hilbert)
            "j": {"ll": [], "lh": [], "hl": [], "hh": []},  # Imaginary part (y-Hilbert)
            "k": {"ll": [], "lh": [], "hl": [], "hh": []},  # Imaginary part (xy-Hilbert)
        }

        # Generate Hilbert transforms of the input
        x_hilbert_x = self._apply_hilbert(x, "x")
        x_hilbert_y = self._apply_hilbert(x, "y")
        x_hilbert_xy = self._apply_hilbert(x, "xy")

        # Initialize with original signals
        ll_r = x
        ll_i = x_hilbert_x
        ll_j = x_hilbert_y
        ll_k = x_hilbert_xy

        # Perform wavelet decomposition for each level
        for i in range(level):
            # Real part decomposition
            ll_r, lh_r, hl_r, hh_r = self._dwt_single_level(ll_r)

            # x-Hilbert part decomposition
            ll_i, lh_i, hl_i, hh_i = self._dwt_single_level(ll_i)

            # y-Hilbert part decomposition
            ll_j, lh_j, hl_j, hh_j = self._dwt_single_level(ll_j)

            # xy-Hilbert part decomposition
            ll_k, lh_k, hl_k, hh_k = self._dwt_single_level(ll_k)

            # Store results for real part
            qwt_coeffs["r"]["ll"].append(ll_r)
            qwt_coeffs["r"]["lh"].append(lh_r)
            qwt_coeffs["r"]["hl"].append(hl_r)
            qwt_coeffs["r"]["hh"].append(hh_r)

            # Store results for x-Hilbert part
            qwt_coeffs["i"]["ll"].append(ll_i)
            qwt_coeffs["i"]["lh"].append(lh_i)
            qwt_coeffs["i"]["hl"].append(hl_i)
            qwt_coeffs["i"]["hh"].append(hh_i)

            # Store results for y-Hilbert part
            qwt_coeffs["j"]["ll"].append(ll_j)
            qwt_coeffs["j"]["lh"].append(lh_j)
            qwt_coeffs["j"]["hl"].append(hl_j)
            qwt_coeffs["j"]["hh"].append(hh_j)

            # Store results for xy-Hilbert part
            qwt_coeffs["k"]["ll"].append(ll_k)
            qwt_coeffs["k"]["lh"].append(lh_k)
            qwt_coeffs["k"]["hl"].append(hl_k)
            qwt_coeffs["k"]["hh"].append(hh_k)

        return qwt_coeffs

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level DWT decomposition."""
        batch, channels, height, width = x.shape
        x = x.view(batch * channels, 1, height, width)

        # Calculate proper padding for the filter size
        filter_size = self.dec_lo.size(0)
        pad_size = filter_size // 2

        # Pad for proper convolution
        try:
            x_pad = F.pad(x, (pad_size,) * 4, mode="reflect")
        except RuntimeError:
            # Fallback for very small tensors
            x_pad = F.pad(x, (pad_size,) * 4, mode="constant")

        # Apply filter to rows
        lo = F.conv2d(x_pad, self.dec_lo.view(1, 1, -1, 1), stride=(2, 1))
        hi = F.conv2d(x_pad, self.dec_hi.view(1, 1, -1, 1), stride=(2, 1))

        # Apply filter to columns
        ll = F.conv2d(lo, self.dec_lo.view(1, 1, 1, -1), stride=(1, 2))
        lh = F.conv2d(lo, self.dec_hi.view(1, 1, 1, -1), stride=(1, 2))
        hl = F.conv2d(hi, self.dec_lo.view(1, 1, 1, -1), stride=(1, 2))
        hh = F.conv2d(hi, self.dec_hi.view(1, 1, 1, -1), stride=(1, 2))

        # Reshape back to batch format
        ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(x.device)
        lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(x.device)
        hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(x.device)
        hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(x.device)

        return ll, lh, hl, hh


class WaveletLoss(nn.Module):
    """Wavelet-based loss calculation module."""

    def __init__(
        self,
        wavelet="db4",
        level=3,
        transform_type="dwt",
        loss_fn: LossCallable = F.mse_loss,
        device=torch.device("cpu"),
        band_level_weights: Optional[dict[str, float]] = None,
        band_weights: Optional[dict[str, float]] = None,
        quaternion_component_weights: dict[str, float] | None = None,
        ll_level_threshold: Optional[int] = -1,
        metrics: bool = False,
        energy_ratio: float = 0.0,
        energy_scale_factor: float = 0.01,
        normalize_bands: bool = True,
        max_timestep: float = 1.0,
        timestep_intensity: float = 0.5,
    ):
        """

        Args:
            wavelet: Wavelet family (e.g., 'db4', 'sym7')
            level: Decomposition level
            transform_type: Type of wavelet transform ('dwt' or 'swt')
            loss_fn: Loss function to apply to wavelet coefficients
            device: Computation device
            band_level_weights: Optional custom weights for different bands on different levels
            band_weights: Optional custom weights for different bands
            component_weights: Weights for quaternion components
            ll_level_threshold: Level when applying loss for ll. Default -1 or last level.
        """
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.transform_type = transform_type
        self.loss_fn = loss_fn
        self.device = device
        self.ll_level_threshold = ll_level_threshold if ll_level_threshold is not None else None
        self.metrics = metrics
        self.energy_ratio = energy_ratio
        self.energy_scale_factor = energy_scale_factor
        self.max_timestep = max_timestep
        self.timestep_intensity = timestep_intensity
        self.normalize_bands = normalize_bands

        # Initialize transform based on type
        if transform_type == "dwt":
            self.transform = DiscreteWaveletTransform(wavelet, device)
        elif transform_type == "swt":  # swt
            self.transform = StationaryWaveletTransform(wavelet, device)
        elif transform_type == "qwt":
            self.transform = QuaternionWaveletTransform(wavelet, device)

            # Register Hilbert filters as buffers
            self.register_buffer("hilbert_x", self.transform.hilbert_x)
            self.register_buffer("hilbert_y", self.transform.hilbert_y)
            self.register_buffer("hilbert_xy", self.transform.hilbert_xy)

            # Default weights
            self.component_weights = quaternion_component_weights or {
                "r": 1.0,  # Real part (standard wavelet)
                "i": 0.7,  # x-Hilbert (imaginary part)
                "j": 0.7,  # y-Hilbert (imaginary part)
                "k": 0.5,  # xy-Hilbert (imaginary part)
            }
        else:
            raise RuntimeError(f"Invalid transform type {transform_type}")

        # Register wavelet filters as module buffers
        self.register_buffer("dec_lo", self.transform.dec_lo.to(device))
        self.register_buffer("dec_hi", self.transform.dec_hi.to(device))

        # Default weights from paper:
        # "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses"
        self.band_level_weights = band_level_weights or {}
        self.band_weights = band_weights or {"ll": 0.1, "lh": 0.01, "hl": 0.01, "hh": 0.05}

    def forward(
        self, pred_latent: Tensor, target_latent: Tensor, timestep: torch.Tensor | None = None
    ) -> tuple[list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate wavelet loss between prediction and target.

        Returns:
            loss: Total wavelet loss
            metrics: Wavelet metrics if requested in WaveletLoss(metrics=True)
        """
        if isinstance(self.transform, QuaternionWaveletTransform):
            return self.quaternion_forward(pred_latent, target_latent)

        batch_size = pred_latent.shape[0]
        device = pred_latent.device

        # Decompose inputs
        pred_coeffs = self.transform.decompose(pred_latent, self.level)
        target_coeffs = self.transform.decompose(target_latent, self.level)

        # Calculate weighted loss
        pattern_losses = []
        combined_hf_pred = []
        combined_hf_target = []
        metrics = {}

        # Use original weights by default
        band_weights = self.band_weights
        band_level_weights = self.band_level_weights

        # Apply timestep-based weighting if provided
        # if timestep is not None:
        #     # Let users control intensity of timestep weighting (0.5 = moderate effect)
        #     intensity = getattr(self, "timestep_intensity", 0.5)
        #     current_band_weights, current_band_level_weights = self.noise_aware_weighting(
        #         timestep, self.max_timestep, intensity=intensity
        #     )

        # If negative it's from the end of the levels else it's the level.
        ll_threshold = None
        if self.ll_level_threshold is not None:
            ll_threshold = self.ll_level_threshold if self.ll_level_threshold > 0 else self.level + self.ll_level_threshold

        # 1. Pattern Loss (using normalization)
        for i in range(self.level):
            pattern_level_losses = torch.zeros_like(pred_coeffs["lh"][i])

            # High frequency bands
            for band in ["ll", "lh", "hl", "hh"]:
                # Skip LL bands except for ones at or beyond the threshold
                if ll_threshold is not None and band == "ll" and i + 1 <= ll_threshold:
                    continue

                weight_key = f"{band}{i+1}"

                if band in pred_coeffs and band in target_coeffs:
                    if self.normalize_bands:
                        # Normalize wavelet components
                        pred_coeffs[band][i] = (pred_coeffs[band][i] - pred_coeffs[band][i].mean()) / (pred_coeffs[band][i].std() + 1e-8)
                        target_coeffs[band][i] = (target_coeffs[band][i] - target_coeffs[band][i].mean()) / (target_coeffs[band][i].std() + 1e-8)

                    weight = band_level_weights.get(weight_key, band_weights[band])
                    band_loss = weight * self.loss_fn(pred_coeffs[band][i], target_coeffs[band][i])
                    pattern_level_losses += band_loss.mean(dim=0)  # mean stack dim

                    # Collect high frequency bands for visualization
                    combined_hf_pred.append(pred_coeffs[band][i])
                    combined_hf_target.append(target_coeffs[band][i])

            pattern_losses.append(pattern_level_losses)

        # TODO: need to update this to work with a list of losses
        # If we are balancing the energy loss with the pattern loss
        # if self.energy_ratio > 0.0:
        #     energy_loss = self.energy_matching_loss(batch_size, pred_coeffs, target_coeffs, device)
        #
        #     loss = (
        #         (1 - self.energy_ratio) * pattern_loss  # Core spatial patterns
        #         + self.energy_ratio * (self.energy_scale_factor * energy_loss)  # Fixes energy disparity
        #     )
        # else:
        energy_loss = None
        losses = pattern_losses

        # METRICS: Calculate all additional metrics (no gradients needed)
        if self.metrics:
            with torch.no_grad():
                # Raw energy metrics
                for band in ["lh", "hl", "hh"]:
                    for i in range(1, self.level + 1):
                        pred_stack = pred_coeffs[band][i - 1]
                        target_stack = target_coeffs[band][i - 1]

                        metrics[f"{band}{i}_raw_pred_energy"] = torch.mean(pred_stack**2).item()
                        metrics[f"{band}{i}_raw_target_energy"] = torch.mean(target_stack**2).item()
                        metrics[f"{band}{i}_energy_ratio"] = (
                            torch.mean(pred_stack**2) / (torch.mean(target_stack**2) + 1e-8)
                        ).item()

                metrics.update(self.calculate_correlation_metrics(pred_coeffs, target_coeffs))
                metrics.update(self.calculate_cross_scale_consistency_metrics(pred_coeffs, target_coeffs))
                metrics.update(self.calculate_directional_consistency_metrics(pred_coeffs, target_coeffs))
                metrics.update(self.calculate_sparsity_metrics(pred_coeffs, target_coeffs))
                metrics.update(self.calculate_latent_regularity_metrics(pred_latent))

                # Add loss components to metrics
                for i, pattern_loss in enumerate(pattern_losses):
                    metrics[f"pattern_loss-{i+1}"] = pattern_loss.detach().mean().item()

                for i, total_loss in enumerate(losses):
                    metrics[f"total_loss-{i+1}"] = total_loss.detach().mean().item()

                if energy_loss is not None:
                    metrics["energy_loss"] = energy_loss.detach().mean().item()

        # Combine high frequency bands for visualization
        if combined_hf_pred and combined_hf_target:
            combined_hf_pred = self._pad_tensors(combined_hf_pred)
            combined_hf_target = self._pad_tensors(combined_hf_target)

            combined_hf_pred = torch.cat(combined_hf_pred, dim=1)
            combined_hf_target = torch.cat(combined_hf_target, dim=1)

            metrics["combined_hf_pred"] = combined_hf_pred.detach().mean().item()
            metrics["combined_hf_target"] = combined_hf_target.detach().mean().item()
        else:
            combined_hf_pred = None
            combined_hf_target = None

        return losses, metrics

    def quaternion_forward(self, pred: Tensor, target: Tensor) -> tuple[list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate QWT loss between prediction and target.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Tuple of (total loss, detailed component losses)
        """
        assert isinstance(self.transform, QuaternionWaveletTransform), "Not a quaternion wavelet transform"
        # Apply QWT to both inputs
        pred_qwt = self.transform.decompose(pred, self.level)
        target_qwt = self.transform.decompose(target, self.level)

        # Initialize total loss and component losses
        total_losses = []
        component_losses = {
            f"{component}_{band}_{level+1}": torch.zeros_like(pred_qwt[component][band][level], device=pred.device)
            for level in range(self.level)
            for component in ["r", "i", "j", "k"]
            for band in ["ll", "lh", "hl", "hh"]
        }

        # Calculate loss for each quaternion component, band and level
        for level_idx in range(self.level):
            pattern_level_losses = torch.zeros_like(pred_qwt["r"]["lh"][level_idx])
            for band in ["ll", "lh", "hl", "hh"]:
                band_weight = self.band_weights[band]
                for component in ["r", "i", "j", "k"]:
                    component_weight = self.component_weights[component]

                    band_level_key = f"{band}{level_idx + 1}"
                    # band_level_weights take priority over band_weight if exists
                    if band_level_key in self.band_level_weights:
                        level_weight = self.band_level_weights[band_level_key]
                    else:
                        level_weight = band_weight

                    # Get coefficients at this level
                    pred_coeff = pred_qwt[component][band][level_idx]
                    target_coeff = target_qwt[component][band][level_idx]

                    # Calculate loss
                    level_loss = self.loss_fn(pred_coeff, target_coeff)

                    # Apply weights
                    weighted_loss = component_weight * level_weight * level_loss

                    # Add to total loss
                    pattern_level_losses += weighted_loss

                    # Add to component loss
                    component_losses[f"{component}_{band}_{level_idx+1}"] += weighted_loss


            total_losses.append(pattern_level_losses)

        metrics = {k: v.detach().mean().item() for k, v in component_losses.items()}
        return total_losses, metrics

    def _pad_tensors(self, tensors: list[Tensor]) -> list[Tensor]:
        """Pad tensors to match the largest size."""
        # Find max dimensions
        max_h = max(t.shape[2] for t in tensors)
        max_w = max(t.shape[3] for t in tensors)

        padded_tensors = []
        for tensor in tensors:
            h_pad = max_h - tensor.shape[2]
            w_pad = max_w - tensor.shape[3]

            if h_pad > 0 or w_pad > 0:
                # Pad bottom and right to match max dimensions
                padded = F.pad(tensor, (0, w_pad, 0, h_pad))
                padded_tensors.append(padded)
            else:
                padded_tensors.append(tensor)

        return padded_tensors

    def energy_matching_loss(
        self, batch_size: int, pred_coeffs: dict[str, list[Tensor]], target_coeffs: dict[str, list[Tensor]], device: torch.device
    ) -> Tensor:
        energy_loss = torch.zeros(batch_size, device=device)
        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level + 1):
                weight_key = f"{band}{i}"
                # Calculate band energies
                pred_energy = torch.mean(pred_coeffs[band][i - 1] ** 2)
                target_energy = torch.mean(target_coeffs[band][i - 1] ** 2)

                # Log-scale energy ratio loss (more stable than direct ratio)
                ratio_loss = torch.abs(torch.log(pred_energy + 1e-8) - torch.log(target_energy + 1e-8))

                weight = self.band_level_weights.get(weight_key, self.band_weights[band])
                energy_loss += weight * ratio_loss

        return energy_loss

    @torch.no_grad()
    def calculate_raw_energy_metrics(self, pred_stack: Tensor, target_stack: Tensor, band: str, level: int):
        metrics: dict[str, float | int] = {}
        metrics[f"{band}{level}_raw_pred_energy"] = torch.mean(pred_stack**2).detach().item()
        metrics[f"{band}{level}_raw_target_energy"] = torch.mean(target_stack**2).detach().item()

        metrics[f"{band}{level}_raw_error"] = self.loss_fn(pred_stack.float(), target_stack.float()).detach().item()

        return metrics

    @torch.no_grad()
    def calculate_cross_scale_consistency_metrics(
        self, pred_coeffs: dict[str, list[Tensor]], target_coeffs: dict[str, list[Tensor]]
    ) -> dict:
        """Calculate metrics for cross-scale consistency"""
        metrics = {}

        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level):
                # Compare ratio of energies between adjacent scales
                pred_energy_fine = torch.mean(pred_coeffs[band][i - 1] ** 2).item()
                pred_energy_coarse = torch.mean(pred_coeffs[band][i] ** 2).item()
                target_energy_fine = torch.mean(target_coeffs[band][i - 1] ** 2).item()
                target_energy_coarse = torch.mean(target_coeffs[band][i] ** 2).item()

                # Calculate ratios and log differences
                pred_ratio = pred_energy_coarse / (pred_energy_fine + 1e-8)
                target_ratio = target_energy_coarse / (target_energy_fine + 1e-8)
                log_ratio_diff = abs(math.log(pred_ratio + 1e-8) - math.log(target_ratio + 1e-8))

                # Store individual metrics
                metrics[f"{band}{i}_to_{i + 1}_pred_scale_ratio"] = pred_ratio
                metrics[f"{band}{i}_to_{i + 1}_target_scale_ratio"] = target_ratio
                metrics[f"{band}{i}_to_{i + 1}_scale_log_diff"] = log_ratio_diff

        # Calculate average difference across all bands and scales
        if metrics:  # Check if dictionary is not empty
            metrics["avg_cross_scale_difference"] = sum(v for k, v in metrics.items() if k.endswith("scale_log_diff")) / len(
                [k for k in metrics if k.endswith("scale_log_diff")]
            )

        return metrics

    @torch.no_grad()
    def calculate_correlation_metrics(self, pred_coeffs: dict[str, list[Tensor]], target_coeffs: dict[str, list[Tensor]]) -> dict:
        """Calculate correlation metrics between prediction and target wavelet coefficients"""
        metrics = {}
        avg_correlations = []

        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level + 1):
                # Get coefficients
                pred = pred_coeffs[band][i - 1]
                target = target_coeffs[band][i - 1]

                # Flatten for batch-wise correlation
                batch_size = pred.shape[0]
                pred_flat = pred.view(batch_size, -1)
                target_flat = target.view(batch_size, -1)

                # Center data
                pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
                target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

                # Calculate correlation
                numerator = torch.sum(pred_centered * target_centered, dim=1)
                denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * torch.sum(target_centered**2, dim=1) + 1e-8)
                correlation = numerator / denominator

                # Average across batch
                avg_correlation = correlation.mean().item()
                metrics[f"{band}{i}_correlation"] = avg_correlation
                avg_correlations.append(avg_correlation)

        # Calculate average correlation across all bands
        if avg_correlations:
            metrics["avg_correlation"] = sum(avg_correlations) / len(avg_correlations)

        return metrics

    @torch.no_grad()
    def calculate_directional_consistency_metrics(
        self, pred_coeffs: dict[str, list[Tensor]], target_coeffs: dict[str, list[Tensor]]
    ) -> dict:
        """Calculate metrics for directional consistency between bands"""
        metrics = {}
        hv_diffs = []
        diag_diffs = []

        for i in range(1, self.level + 1):
            # Horizontal to vertical energy ratio
            pred_hl_energy = torch.mean(pred_coeffs["hl"][i - 1] ** 2).item()
            pred_lh_energy = torch.mean(pred_coeffs["lh"][i - 1] ** 2).item()
            target_hl_energy = torch.mean(target_coeffs["hl"][i - 1] ** 2).item()
            target_lh_energy = torch.mean(target_coeffs["lh"][i - 1] ** 2).item()

            pred_hv_ratio = pred_hl_energy / (pred_lh_energy + 1e-8)
            target_hv_ratio = target_hl_energy / (target_lh_energy + 1e-8)
            hv_log_diff = abs(math.log(pred_hv_ratio + 1e-8) - math.log(target_hv_ratio + 1e-8))

            # Diagonal to (horizontal+vertical) energy ratio
            pred_hh_energy = torch.mean(pred_coeffs["hh"][i - 1] ** 2).item()
            target_hh_energy = torch.mean(target_coeffs["hh"][i - 1] ** 2).item()

            pred_d_ratio = pred_hh_energy / (pred_hl_energy + pred_lh_energy + 1e-8)
            target_d_ratio = target_hh_energy / (target_hl_energy + target_lh_energy + 1e-8)
            diag_log_diff = abs(math.log(pred_d_ratio + 1e-8) - math.log(target_d_ratio + 1e-8))

            # Store metrics
            metrics[f"level{i}_horiz_vert_pred_ratio"] = pred_hv_ratio
            metrics[f"level{i}_horiz_vert_target_ratio"] = target_hv_ratio
            metrics[f"level{i}_horiz_vert_log_diff"] = hv_log_diff

            metrics[f"level{i}_diag_ratio_pred"] = pred_d_ratio
            metrics[f"level{i}_diag_ratio_target"] = target_d_ratio
            metrics[f"level{i}_diag_ratio_log_diff"] = diag_log_diff

            hv_diffs.append(hv_log_diff)
            diag_diffs.append(diag_log_diff)

        # Average metrics
        if hv_diffs:
            metrics["avg_horiz_vert_diff"] = sum(hv_diffs) / len(hv_diffs)
        if diag_diffs:
            metrics["avg_diag_ratio_diff"] = sum(diag_diffs) / len(diag_diffs)

        return metrics

    @torch.no_grad()
    def calculate_latent_regularity_metrics(self, pred_latents: Tensor) -> dict:
        """Calculate metrics for latent space regularity"""
        metrics = {}

        # Calculate gradient magnitude of latent representation
        grad_x = pred_latents[:, :, 1:, :] - pred_latents[:, :, :-1, :]
        grad_y = pred_latents[:, :, :, 1:] - pred_latents[:, :, :, :-1]

        # Total variation
        tv_x = torch.mean(torch.abs(grad_x)).item()
        tv_y = torch.mean(torch.abs(grad_y)).item()
        tv_total = tv_x + tv_y

        # Statistical metrics
        std_value = torch.std(pred_latents).item()
        mean_value = torch.mean(pred_latents).item()
        std_diff = abs(std_value - 1.0)

        # Store metrics
        metrics["latent_tv_x"] = tv_x
        metrics["latent_tv_y"] = tv_y
        metrics["latent_tv_total"] = tv_total
        metrics["latent_std"] = std_value
        metrics["latent_mean"] = mean_value
        metrics["latent_std_from_normal"] = std_diff

        return metrics

    @torch.no_grad()
    def calculate_sparsity_metrics(
        self, coeffs: dict[str, list[Tensor]], reference_coeffs: dict[str, list[Tensor]] | None = None
    ) -> dict:
        """Calculate sparsity metrics for wavelet coefficients"""
        metrics = {}
        band_sparsities = []

        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level + 1):
                coef = coeffs[band][i - 1]

                # L1 norm (sparsity measure)
                l1_norm = torch.mean(torch.abs(coef)).item()
                metrics[f"{band}{i}_l1_norm"] = l1_norm
                band_sparsities.append(l1_norm)

                # Additional sparsity metrics
                non_zero_ratio = torch.mean((torch.abs(coef) > 0.01).float()).item()
                metrics[f"{band}{i}_non_zero_ratio"] = non_zero_ratio

                # If reference coefficients provided, calculate relative sparsity
                if reference_coeffs is not None:
                    ref_coef = reference_coeffs[band][i - 1]
                    ref_l1_norm = torch.mean(torch.abs(ref_coef)).item()
                    rel_sparsity = l1_norm / (ref_l1_norm + 1e-8)
                    metrics[f"{band}{i}_relative_sparsity"] = rel_sparsity

        # Average sparsity across bands
        if band_sparsities:
            metrics["avg_l1_sparsity"] = sum(band_sparsities) / len(band_sparsities)

        return metrics

    # TODO: does not work right in terms of weighting in an appropriate range
    def noise_aware_weighting(self, timestep: Tensor, max_timestep: float, intensity=1.0):
        """
        Adjust band weights based on diffusion timestep, maintaining reasonable magnitudes

        Args:
            timestep: Current diffusion timestep
            max_timestep: Maximum diffusion timestep
            intensity: Controls how strongly timestep affects weights (0.0-1.0)

        Returns:
            Dictionary of adjusted weights with reasonable magnitudes
        """
        # Calculate denoising progress (0.0 = noisy start, 1.0 = clean end)
        progress = 1.0 - (timestep / max_timestep)

        # Initialize adjusted weights dictionaries
        band_weights_adjusted = {}
        band_level_weights_adjusted = {}

        # Define target ranges for weights
        # These ensure weights stay within reasonable bounds regardless of input
        ll_range = (0.5, 2.0)  # Low-frequency weights
        hf_range = (0.01, 0.2)  # High-frequency weights (lh, hl)
        hh_range = (0.005, 0.1)  # Diagonal details weight (hh)

        # Determine sign for each weight - properly handling different types
        def get_sign(w):
            if isinstance(w, torch.Tensor):
                # For tensor weights: check if all values are positive
                if w.numel() > 1:
                    return 1 if (w > 0).all().item() else -1
                else:
                    return 1 if w.item() > 0 else -1
            else:
                # For float or int weights
                return 1 if w > 0 else -1

        # Get sign of each band weight (to preserve positive/negative direction)
        signs = {band: get_sign(weight) for band, weight in self.band_weights.items()}

        # Apply modulated weighting based on progress
        for band, weight in self.band_weights.items():
            if band == "ll":
                # For low frequency: high at start, decreases toward end
                # Map from progress to target range
                target_value = ll_range[0] + (1.0 - progress) * (ll_range[1] - ll_range[0]) * intensity
            elif band == "hh":
                # For diagonal details: low at start, increases toward end
                target_value = hh_range[0] + progress * (hh_range[1] - hh_range[0]) * intensity
            else:  # "lh", "hl"
                # For horizontal/vertical details: low at start, increases toward end
                target_value = hf_range[0] + progress * (hf_range[1] - hf_range[0]) * intensity

            # Apply sign to preserve direction
            target_value = target_value * signs[band]

            # Calculate blend factor - how much of original vs. target weight to use
            # Higher intensity means more influence from the target values
            blend_factor = min(intensity, 0.8)  # Cap at 0.8 to preserve some original weight

            # Create tamed weight by blending original (normalized) and target values
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                # Handle tensor weights (multiple values)
                weight_mean = torch.abs(weight).mean()
                normalized_weight = weight / (weight_mean + 1e-8)
                # Blend between normalized weight and target
                blended_weight = (1 - blend_factor) * normalized_weight + blend_factor * target_value
                band_weights_adjusted[band] = blended_weight
            else:
                # Handle scalar weights
                weight_abs = abs(weight) if isinstance(weight, (int, float)) else abs(weight.item())
                normalized_weight = weight / (weight_abs + 1e-8)
                # Blend between normalized weight and target
                blended_weight = (1 - blend_factor) * normalized_weight + blend_factor * target_value
                band_weights_adjusted[band] = blended_weight

        # Similar approach for band_level_weights
        for key, weight in self.band_level_weights.items():
            band = key[:2]  # Extract band name (e.g., "ll" from "ll1")
            level = int(key[2:])  # Extract level number

            # Determine appropriate target range based on band and level
            if band == "ll":
                # Low frequency bands: higher weight early
                level_factor = level / self.level  # Lower levels have lower factor
                target_range = (ll_range[0] * (1 - level_factor), ll_range[1] * (1 - 0.3 * level_factor))
                target_value = target_range[0] + (1.0 - progress) * (target_range[1] - target_range[0]) * intensity
            elif band == "hh":
                # Diagonal details: lower weight early
                level_factor = (self.level - level + 1) / self.level  # Higher levels have higher factor
                target_range = (hh_range[0] * level_factor, hh_range[1] * level_factor)
                target_value = target_range[0] + progress * (target_range[1] - target_range[0]) * intensity
            else:  # "lh", "hl"
                # Horizontal/vertical details: lower weight early
                level_factor = (self.level - level + 1) / self.level  # Higher levels have higher factor
                target_range = (hf_range[0] * level_factor, hf_range[1] * level_factor)
                target_value = target_range[0] + progress * (target_range[1] - target_range[0]) * intensity

            # Apply sign to preserve direction
            sign = 1 if weight > 0 else -1
            target_value = target_value * sign

            # Calculate blend factor
            blend_factor = min(intensity, 0.8)

            # Create tamed weight
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                weight_mean = torch.abs(weight).mean()
                normalized_weight = weight / (weight_mean + 1e-8)
                blended_weight = (1 - blend_factor) * normalized_weight + blend_factor * target_value
            else:
                weight_abs = abs(weight) if isinstance(weight, (int, float)) else abs(weight.item())
                normalized_weight = weight / (weight_abs + 1e-8)
                blended_weight = (1 - blend_factor) * normalized_weight + blend_factor * target_value

            band_level_weights_adjusted[key] = blended_weight

        return band_weights_adjusted, band_level_weights_adjusted

    def set_loss_fn(self, loss_fn: LossCallable):
        """
        Set loss function to use. Wavelet loss wants l1 or huber loss.
        """
        self.loss_fn = loss_fn


def visualize_qwt_results(qwt_transform, lr_image, pred_latent, target_latent, filename):
    """
    Visualize QWT decomposition of input, prediction, and target.

    visualize_qwt_results(
        model.qwt_loss.transform,
        lr_images[0:1],
        pred_latents[0:1],
        target_latents[0:1],
        f"qwt_vis_epoch{epoch}_batch{batch_idx}.png"
    )

    Args:
        qwt_transform: Quaternion Wavelet Transform instance
        lr_image: Low-resolution input image
        pred_latent: Predicted latent
        target_latent: Target latent
        filename: Output filename
    """
    import matplotlib.pyplot as plt

    # Apply QWT
    lr_qwt = qwt_transform.decompose(lr_image, level=2)
    pred_qwt = qwt_transform.decompose(pred_latent, level=2)
    target_qwt = qwt_transform.decompose(target_latent, level=2)

    # Set up figure
    fig, axes = plt.subplots(4, 9, figsize=(27, 12))

    # First, show original images/latents
    axes[0, 0].imshow(lr_image[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 0].set_title("LR Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_latent[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 1].set_title("Pred Latent")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(target_latent[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 2].set_title("Target Latent")
    axes[0, 2].axis("off")

    # Keep track of current column
    col = 3

    # For each component (r, i, j, k)
    for i, component in enumerate(["r", "i", "j", "k"]):
        # For first level only, display LL band
        if i == 0:  # Only for real component to save space
            # First level LL band
            lr_ll = lr_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()
            pred_ll = pred_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()
            target_ll = target_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()

            # Normalize for visualization
            lr_ll = (lr_ll - lr_ll.min()) / (lr_ll.max() - lr_ll.min() + 1e-8)
            pred_ll = (pred_ll - pred_ll.min()) / (pred_ll.max() - pred_ll.min() + 1e-8)
            target_ll = (target_ll - target_ll.min()) / (target_ll.max() - target_ll.min() + 1e-8)

            axes[0, col].imshow(lr_ll, cmap="viridis")
            axes[0, col].set_title(f"LR {component}_LL")
            axes[0, col].axis("off")

            axes[0, col + 1].imshow(pred_ll, cmap="viridis")
            axes[0, col + 1].set_title(f"Pred {component}_LL")
            axes[0, col + 1].axis("off")

            axes[0, col + 2].imshow(target_ll, cmap="viridis")
            axes[0, col + 2].set_title(f"Target {component}_LL")
            axes[0, col + 2].axis("off")

            col = 0  # Reset column for next row

        # For each component, show detail bands
        for band_idx, band in enumerate(["lh", "hl", "hh"]):
            # Get band coefficients
            lr_band = lr_qwt[component][band][0][0, 0].detach().cpu().numpy()
            pred_band = pred_qwt[component][band][0][0, 0].detach().cpu().numpy()
            target_band = target_qwt[component][band][0][0, 0].detach().cpu().numpy()

            # Normalize for visualization
            lr_band = (lr_band - lr_band.min()) / (lr_band.max() - lr_band.min() + 1e-8)
            pred_band = (pred_band - pred_band.min()) / (pred_band.max() - pred_band.min() + 1e-8)
            target_band = (target_band - target_band.min()) / (target_band.max() - target_band.min() + 1e-8)

            # Plot in the corresponding row
            row = i + 1 if i > 0 else i + 1 + band_idx

            axes[row, col].imshow(lr_band, cmap="viridis")
            axes[row, col].set_title(f"LR {component}_{band}")
            axes[row, col].axis("off")
            axes[row, col + 1].imshow(pred_band, cmap="viridis")
            axes[row, col + 1].set_title(f"Pred {component}_{band}")
            axes[row, col + 1].axis("off")

            axes[row, col + 2].imshow(target_band, cmap="viridis")
            axes[row, col + 2].set_title(f"Target {component}_{band}")
            axes[row, col + 2].axis("off")

            col += 3

            # Reset column for next row
            if col >= 9:
                col = 0

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


"""
##########################################
# Perlin Noise
def rand_perlin_2d(device, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)),
            dim=-1,
        )
        % 1
    )
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return 1.414 * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(device, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(device, shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def perlin_noise(noise, device, octaves):
    _, c, w, h = noise.shape
    perlin = lambda: rand_perlin_2d_octaves(device, (w, h), (4, 4), octaves)
    noise_perlin = []
    for _ in range(c):
        noise_perlin.append(perlin())
    noise_perlin = torch.stack(noise_perlin).unsqueeze(0)   # (1, c, w, h)
    noise += noise_perlin # broadcast for each batch
    return noise / noise.std()  # Scaled back to roughly unit variance
"""
