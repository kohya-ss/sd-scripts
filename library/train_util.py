# common functions for training

import argparse
import hashlib
import importlib
import math
import pathlib
import shutil
import subprocess
import time
from io import BytesIO
from typing import Optional, Union

import cv2
import platform
import diffusers
import numpy as np
import safetensors.torch
import torch
import transformers
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from einops import rearrange
from torch import einsum
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
from library.dataset.common import KohyaException
from library.loaders import load_models_from_stable_diffusion_checkpoint, load_vae
from library.savers import save_stable_diffusion_checkpoint, save_diffusers_checkpoint

TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


# , ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]         # Linux?


def debug_dataset(train_dataset, show_input_ids=False):
    print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    print("Escape for exit. / Escキーで中断、終了します")

    train_dataset.set_current_epoch(1)
    k = 0
    for i, example in enumerate(train_dataset):
        if example['latents'] is not None:
            print(f"sample has latents from npz file: {example['latents'].size()}")
        for j, (ik, cap, lw, iid) in enumerate(
                zip(example['image_keys'], example['captions'], example['loss_weights'], example['input_ids'])):
            print(f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}"')
            if show_input_ids:
                print(f"input ids: {iid}")
            if example['images'] is not None:
                im = example['images'][j]
                print(f"image size: {im.size()}")
                im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)
                if platform.system() == 'Windows':  # only windows
                    cv2.imshow("img", im)
                k = cv2.waitKey()
                cv2.destroyAllWindows()
                if k == 27:
                    break
        if k == 27 or (example['images'] is None and i >= 8):
            break


# endregion


# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6


# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
  save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=pathlib.Path(__file__).resolve().parent).decode('ascii').strip()
    except:
        return "(unknown)"


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 2 in the paper """

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = (q.shape[-1] ** -0.5)

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_(
                    (exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 4 in the paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2)
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.)

                p = exp_attn_weights / lc

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
    if mem_eff_attn:
        replace_unet_cross_attn_to_memory_efficient()
    elif xformers:
        replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
    print("Replace CrossAttention.forward to use FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')

        # diffusers 0.7.0~  わざわざ変えるなよ (;´Д｀)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
    print("Replace CrossAttention.forward to use xformers")
    try:
        import xformers.ops
    except ImportError:
        raise ImportError("No xformers / xformersがインストールされていないようです")

    def forward_xformers(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)

        context = default(context, x)
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

        out = rearrange(out, 'b n h d -> b n (h d)', h=h)

        # diffusers 0.7.0~
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_xformers


# endregion


# region arguments

def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument("--v2", action='store_true',
                        help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
    parser.add_argument("--v_parameterization", action='store_true',
                        help='enable v-parameterization training / v-parameterization学習を有効にする')
    parser.add_argument("--pretrained_model_name_or_path", type=pathlib.Path, default=None,
                        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / "
                             "学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--optimizer_type", type=str, default="",
                        help="Optimizer to use / "
                             "オプティマイザの種類: AdamW (default), AdamW8bit, Lion, SGDNesterov, "
                             "SGDNesterov8bit, DAdaptation, AdaFactor")

    # backward compatibility
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer (requires bitsandbytes) / "
                             "8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）")

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない")

    parser.add_argument("--optimizer_args", type=str, default=None, nargs='*',
                        help="additional arguments for optimizer (like \"weight_decay=0.01 betas=0.9,0.999 ...\") / "
                             "オプティマイザの追加引数（例： \"weight_decay=0.01 betas=0.9,0.999 ...\"）")

    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="scheduler to use for learning rate / "
                             "学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0) / "
                             "学習率のスケジューラをウォームアップするステップ数（デフォルト0）")
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / "
                             "cosine with restartsスケジューラでのリスタート回数")
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power")


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
    parser.add_argument("--output_name", type=str, default=None,
                        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    parser.add_argument("--save_precision", type=str, default=None,
                        choices=[None, "float", "fp16", "bf16"], help="precision in saving / 保存時に精度を変更して保存する")
    parser.add_argument("--save_every_n_epochs", type=int, default=None,
                        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
    parser.add_argument("--save_n_epoch_ratio", type=int, default=None,
                        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / "
                             "学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）")
    parser.add_argument("--save_last_n_epochs", type=int, default=None, help="save last N checkpoints / 最大Nエポック保存する")
    parser.add_argument("--save_last_n_epochs_state", type=int, default=None,
                        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ "
                             "最大Nエポックstateを保存する(--save_last_n_epochsの指定を上書きします)")
    parser.add_argument("--save_state", action="store_true",
                        help="save training state additionally (including optimizer states etc.) / "
                             "optimizerなど学習状態も含めたstateを追加で保存する")
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / "
                             "text encoderのトークンの最大長（未指定で75、150または225が指定可）")
    parser.add_argument("--mem_eff_attn", action="store_true",
                        help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
    parser.add_argument("--xformers", action="store_true",
                        help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument("--vae", type=str, default=None,
                        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument("--max_train_epochs", type=int, default=None,
                        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=8,
                        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / "
                             "DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）")
    parser.add_argument("--persistent_data_loader_workers", action="store_true",
                        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / "
                             "DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="enable gradient checkpointing / grandient checkpointingを有効にする")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass / "
                             "学習時に逆伝播をする前に勾配を合計するステップ数")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1) / "
                             "text encoderの後ろからn番目の層の出力を用いる（nは1以上）")
    parser.add_argument("--logging_dir", type=pathlib.Path, default=None,
                        help="enable logging and output TensorBoard log to this directory / "
                             "ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する")
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    parser.add_argument("--noise_offset", type=float, default=None,
                        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / "
                             "Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）")
    parser.add_argument("--lowram", action="store_true",
                        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / "
                             "メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込むなど（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）")

    if support_dreambooth:
        # DreamBooth training
        parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                            help="loss weight for regularization images / 正則化画像のlossの重み")


def verify_training_args(args: argparse.Namespace):
    if args.v_parameterization and not args.v2:
        print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
    if args.v2 and args.clip_skip is not None:
        print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")


def add_dataset_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool,
                          support_caption_dropout: bool):
    # dataset common
    parser.add_argument("--train_data_dir", type=pathlib.Path, default=None, help="directory for train images / "
                                                                         "学習画像データのディレクトリ")
    parser.add_argument("--shuffle_caption", action="store_true",
                        help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
    parser.add_argument("--caption_extension", type=str, default=".caption",
                        help="extension of caption files / 読み込むcaptionファイルの拡張子")
    parser.add_argument("--caption_extention", type=str, default=None,
                        help="extension of caption files (backward compatibility) / "
                             "読み込むcaptionファイルの拡張子（スペルミスを残してあります）")
    parser.add_argument("--keep_tokens", type=int, default=None,
                        help="keep heading N tokens when shuffling caption tokens / "
                             "captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す")
    parser.add_argument("--color_aug", action="store_true",
                        help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    parser.add_argument("--flip_aug", action="store_true",
                        help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    parser.add_argument("--face_crop_aug_range", type=str, default=None,
                        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / "
                             "学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）")
    parser.add_argument("--random_crop", action="store_true",
                        help="enable random crop (for style training in face-centered crop augmentation) / "
                             "ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）")
    parser.add_argument("--debug_dataset", action="store_true",
                        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）")
    parser.add_argument("--resolution", type=str, default=None,
                        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）")
    parser.add_argument("--cache_latents", action="store_true",
                        help="cache latents to reduce memory (augmentations must be disabled) / "
                             "メモリ削減のためにlatentをcacheする（augmentationは使用不可）")
    parser.add_argument("--enable_bucket", action="store_true",
                        help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument("--bucket_reso_steps", type=int, default=64,
                        help="steps of resolution for buckets, divisible by 8 is recommended / "
                             "bucketの解像度の単位、8で割り切れる値を推奨します")
    parser.add_argument("--bucket_no_upscale", action="store_true",
                        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します")

    if support_caption_dropout:
        # Textual Inversion はcaptionのdropoutをsupportしない
        # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
        parser.add_argument("--caption_dropout_rate", type=float, default=0,
                            help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合")
        parser.add_argument("--caption_dropout_every_n_epochs", type=int, default=None,
                            help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする")
        parser.add_argument("--caption_tag_dropout_rate", type=float, default=0,
                            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合")

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument("--reg_data_dir", type=pathlib.Path, default=None,
                            help="directory for regularization images / 正則化画像データのディレクトリ")

    if support_caption:
        # caption dataset
        parser.add_argument("--in_json", type=str, default=None,
                            help="json metadata for dataset / データセットのmetadataのjsonファイル")
        parser.add_argument("--dataset_repeats", type=int, default=1,
                            help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数")


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--save_model_as", type=str, default=None,
                        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
                        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）")
    parser.add_argument("--use_safetensors", action='store_true',
                        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）")


# endregion

# region utils


def get_optimizer(args, trainable_params):
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation, Adafactor"

    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert not args.use_lion_optimizer, "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
        assert optimizer_type is None or optimizer_type == "", "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "AdamW8bit"

    elif args.use_lion_optimizer:
        assert optimizer_type is None or optimizer_type == "", "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    # 引数を分解する：boolとfloat、tupleのみ対応
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split('=')

            value = value.split(",")
            for i in range(len(value)):
                if value[i].lower() == "true" or value[i].lower() == "false":
                    value[i] = (value[i].lower() == "true")
                else:
                    value[i] = float(value[i])
            if len(value) == 1:
                value = value[0]
            else:
                value = tuple(value)

            optimizer_kwargs[key] = value
    # print("optkwargs:", optimizer_kwargs)

    lr = args.learning_rate

    if optimizer_type == "AdamW8bit".lower():
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
        print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = bnb.optim.AdamW8bit
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov8bit".lower():
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
        print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(
                f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = bnb.optim.SGD8bit
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
        print(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(
                f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type == "DAdaptation".lower():
        try:
            import dadaptation
        except ImportError:
            raise ImportError("No dadaptation / dadaptation がインストールされていないようです")
        print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")

        min_lr = lr
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            for group in trainable_params:
                min_lr = min(min_lr, group.get("lr", lr))

        if min_lr <= 0.1:
            print(
                f'learning rate is too low. If using dadaptation, set learning rate around 1.0 / 学習率が低すぎるようです。1.0前後の値を指定してください: {min_lr}')
            print('recommend option: lr=1.0 / 推奨は1.0です')

        optimizer_class = dadaptation.DAdaptAdam
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
            optimizer_kwargs["relative_step"] = True
        print(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
            args.learning_rate = None

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    # 一応argsを無効にしておく TODO 依存関係が逆転してるのであまり望ましくない
                    print(f"unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
                    args.unet_lr = None
                    args.text_encoder_lr = None

            if args.lr_scheduler != "adafactor":
                print(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
            args.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません")
            if args.lr_scheduler != "constant_with_warmup":
                print(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                print(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    else:
        # 任意のoptimizerを使う
        optimizer_type = args.optimizer_type  # lowerでないやつ（微妙）
        print(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    return optimizer_name, optimizer_args, optimizer


# Monkeypatch newer get_scheduler() function overridng current version of diffusers.optimizer.get_scheduler
# code is taken from https://github.com/huggingface/diffusers diffusers.optimizer, commit d87cc15977b87160c30abaace3894e802ad9e1e6
# Which is a newer release of diffusers than currently packaged with sd-scripts
# This code can be removed when newer diffusers version (v0.12.1 or greater) is tested and implemented to sd-scripts


def get_scheduler_fix(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
        num_cycles: int = 1,
        power: float = 1.0,
):
    """
  Unified API to get any scheduler from its name.
  Args:
      name (`str` or `SchedulerType`):
          The name of the scheduler to use.
      optimizer (`torch.optim.Optimizer`):
          The optimizer that will be used during training.
      num_warmup_steps (`int`, *optional*):
          The number of warmup steps to do. This is not required by all schedulers (hence the argument being
          optional), the function will raise an error if it's unset and the scheduler type requires it.
      num_training_steps (`int``, *optional*):
          The number of training steps to do. This is not required by all schedulers (hence the argument being
          optional), the function will raise an error if it's unset and the scheduler type requires it.
      num_cycles (`int`, *optional*):
          The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
      power (`float`, *optional*, defaults to 1.0):
          Power factor. See `POLYNOMIAL` scheduler
      last_epoch (`int`, *optional*, defaults to -1):
          The index of the last epoch when resuming training.
  """
    if name.startswith("adafactor"):
        assert type(
            optimizer) == transformers.optimization.Adafactor, f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(':')[1])
        # print("adafactor scheduler init lr", initial_lr)
        return transformers.optimization.AdafactorSchedule(optimizer, initial_lr)

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    if args.cache_latents:
        assert not args.color_aug, "when caching latents, color_aug cannot be used / latentをキャッシュするときはcolor_augは使えません"
        assert not args.random_crop, "when caching latents, random_crop cannot be used / latentをキャッシュするときはrandom_cropは使えません"

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(',')])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert len(args.resolution) == 2, \
            f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(',')])
        assert len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1], \
            f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            print(
                f"latents in npz is ignored when color_aug or random_crop is True / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます")


def load_tokenizer(args: argparse.Namespace):
    print("prepare tokenizer")
    if args.v2:
        tokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
    else:
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
    if args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")
    return tokenizer


def prepare_accelerator(args: argparse.Namespace):
    if args.logging_dir is None:
        log_with = None
        logging_dir = None
    else:
        log_with = "tensorboard"
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime('%Y%m%d%H%M%S', time.localtime())

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=log_with, logging_dir=logging_dir)

    # accelerateの互換性問題を解決する
    accelerator_0_15 = True
    try:
        accelerator.unwrap_model("dummy", True)
        print("Using accelerator 0.15.0 or above.")
    except TypeError:
        accelerator_0_15 = False

    def unwrap_model(model):
        if accelerator_0_15:
            return accelerator.unwrap_model(model, True)
        return accelerator.unwrap_model(model)

    return accelerator, unwrap_model


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def load_target_model(args: argparse.Namespace, weight_dtype):
    name_or_path = pathlib.Path(args.pretrained_model_name_or_path).resolve()
    load_stable_diffusion_format = name_or_path.is_file()  # determine SD or Diffusers
    if load_stable_diffusion_format:
        print("load StableDiffusion checkpoint")
        text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(args.v2, name_or_path)
    else:
        print("load Diffusers pretrained models")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            raise KohyaException(f"Model either doesn't exist as a file or on HuggingFace.co. Invalid filename? / "
                                 f"指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。"
                                 f"ファイル名が誤っているかもしれません: "
                                 f"{name_or_path}")
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

    # VAEを読み込む
    if args.vae is not None:
        vae = load_vae(args.vae, weight_dtype)
        print("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        if args.v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(
                    encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def get_epoch_ckpt_name(args: argparse.Namespace, use_safetensors, epoch):
    model_name = DEFAULT_EPOCH_NAME if args.output_name is None else args.output_name
    ckpt_name = EPOCH_FILE_NAME.format(model_name, epoch) + (".safetensors" if use_safetensors else ".ckpt")
    return model_name, ckpt_name


def save_on_epoch_end(args: argparse.Namespace, save_func, remove_old_func, epoch_no: int, num_train_epochs: int):
    saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
    if saving:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_func()
        if args.save_last_n_epochs is not None:
            remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
            remove_old_func(remove_epoch_no)
    return saving


def save_sd_model_on_epoch_end(args: argparse.Namespace, accelerator, src_path: str, save_stable_diffusion_format: bool,
                               use_safetensors: bool, save_dtype: torch.dtype, epoch: int, num_train_epochs: int,
                               global_step: int, text_encoder, unet, vae):
    epoch_no = epoch + 1
    model_name, ckpt_name = get_epoch_ckpt_name(args, use_safetensors, epoch_no)

    if save_stable_diffusion_format:
        def save_sd():
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            print(f"saving checkpoint: {ckpt_file}")
            save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                             src_path, epoch_no, global_step, save_dtype, vae)

        def remove_sd(old_epoch_no):
            _, old_ckpt_name = get_epoch_ckpt_name(args, use_safetensors, old_epoch_no)
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        save_func = save_sd
        remove_old_func = remove_sd
    else:
        def save_du():
            out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
            print(f"saving model: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                      src_path, vae=vae, use_safetensors=use_safetensors)

        def remove_du(old_epoch_no):
            out_dir_old = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, old_epoch_no))
            if os.path.exists(out_dir_old):
                print(f"removing old model: {out_dir_old}")
                shutil.rmtree(out_dir_old)

        save_func = save_du
        remove_old_func = remove_du

    saving = save_on_epoch_end(args, save_func, remove_old_func, epoch_no, num_train_epochs)
    if saving and args.save_state:
        save_state_on_epoch_end(args, accelerator, model_name, epoch_no)


def save_state_on_epoch_end(args: argparse.Namespace, accelerator, model_name, epoch_no):
    print("saving state.")
    accelerator.save_state(os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            print(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_sd_model_on_train_end(args: argparse.Namespace, src_path: str, save_stable_diffusion_format: bool,
                               use_safetensors: bool, save_dtype: torch.dtype, epoch: int, global_step: int,
                               text_encoder, unet, vae):
    model_name = DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                         src_path, epoch, global_step, save_dtype, vae)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"save trained model as Diffusers to {out_dir}")
        save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                  src_path, vae=vae, use_safetensors=use_safetensors)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    print("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name
    accelerator.save_state(os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name)))


# endregion

# region 前処理用


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return tensor_pil, img_path

# endregion
