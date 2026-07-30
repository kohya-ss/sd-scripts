# Anima ControlNet-LLLite training script
# (anima_train.py を派生し、DiT を凍結して LLLite のみを学習する)

import argparse
import copy
import gc
import math
import os
from multiprocessing import Value
from typing import Optional

# bucket 切替で発生しうる稀な断片化 OOM 対策
# torch import より前に環境変数を設定する必要があるため、ここで setdefault しておく.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import toml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from library import flux_train_utils, qwen_image_autoencoder_kl
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import (
    deepspeed_utils,
    anima_train_utils,
    anima_utils,
    strategy_base,
    strategy_anima,
    sai_model_spec,
)
import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.compile_utils as compile_utils
import library.dataset as dataset_util
import library.model_io as model_io
import library.optimizer as optimizer_util
import library.logging_util as logging_util
import library.loss as loss_util
import library.checkpoint_io as checkpoint_io
import library.sampling as sampling
import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments

import networks.control_net_lllite_anima as lllite_module
from networks.control_net_lllite_anima import (
    ControlNetLLLiteDiT,
    AnimaControlNetLLLiteWrapper,
    encode_reference_hidden_states,
    build_uncond_ref_context,
    install_ref_context_dispatch,
    save_lllite_model,
    load_lllite_weights,
    build_cond_tensors,
    LLLITE_ARCH_VERSION,
    LLLITE_ARCH_VERSION_SEMANTIC,
    COND_INPUT_SPACES as LLLITE_COND_INPUT_SPACES,
    TRUNK_TYPES as LLLITE_TRUNK_TYPES,
    REF_CONTEXT_MODES as LLLITE_REF_CONTEXT_MODES,
    GATE_MODES as LLLITE_GATE_MODES,
    PRESETS as LLLITE_PRESETS,
    ATOMIC_SPECIFIERS as LLLITE_ATOMIC_SPECIFIERS,
)
from library.mask_generator import random_mask as _gen_random_mask

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _load_control_image(path: str, width: int, height: int, device, dtype) -> torch.Tensor:
    """Load a control image and return (1, 3, H, W) in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # HWC, [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
    return tensor.to(device=device, dtype=dtype)


def _load_mask_image(path: str, width: int, height: int, device, dtype) -> torch.Tensor:
    """Load a mask image and return (1, 1, H, W) in {0, 1}.
    1.0 = inpaint area (穴), 0.0 = keep.
    """
    img = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, H, W)
    return tensor.to(device=device, dtype=dtype)


def _generate_random_masks_for_batch(
    batch_size: int, height: int, width: int, device, dtype
) -> torch.Tensor:
    """library.mask_generator.random_mask を使ってバッチ分のランダム mask を生成する.
    返り値: (B, 1, H, W) in {0, 1} (1=inpaint, 0=keep)."""
    masks_np = np.empty((batch_size, 1, height, width), dtype=np.float32)
    for i in range(batch_size):
        pil = _gen_random_mask(width, height)  # PIL "L", 0=keep / 255=inpaint
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr >= 0.5).astype(np.float32)
        masks_np[i, 0] = arr
    return torch.from_numpy(masks_np).to(device=device, dtype=dtype)


def _make_lllite_sample_hooks(
    args,
    lllite,
    dit_dtype,
    vae=None,
    dit=None,
    tokenize_strategy=None,
    text_encoding_strategy=None,
    text_encoder=None,
    sample_prompts_te_outputs=None,
):
    """Build (on_prompt_start, on_prompt_end) callbacks that wire control image / multiplier
    into the LLLite module before each sample prompt is rendered. The pre-sample multiplier is
    saved and restored so that, e.g., `--am 0` for inspection does not leak into training (which
    would otherwise hit the multiplier==0 short-circuit in LLLiteModuleDiT.forward and break
    backward by yielding a graph with no grad).

    In inpainting mode (cond_in_channels=4) the prompt line additionally accepts `--mk <path>`
    for the mask image. If the mask is missing/not found the prompt is rendered without LLLite cond
    (warning logged), so users can intentionally inspect the base DiT.

    `vae` is required when the LLLite runs in latent cond input space (the control image is VAE
    encoded before being fed to conditioning1).

    ref_context='caption' では正負プロンプトの context をここでエンコードし、CFG の各ブランチに
    対応する h_ref を install_ref_context_dispatch で切り替える (学習の dropout 整合と一致)。
    text_encoder 系の引数はそのために使う (zero/uncond では不要)。
    """

    is_inpaint = lllite.cond_in_channels == 4
    is_latent = lllite.cond_input_space == "latent"
    is_semantic = lllite.trunk == "semantic"
    ref_context = lllite.ref_context if is_semantic else "zero"
    assert not is_latent or vae is not None, "vae is required for latent cond input space"
    assert not is_semantic or dit is not None, "dit is required for the semantic trunk (reference forward)"

    saved = {"multiplier": None, "dispatch_handle": None}

    def _encode_prompt_context(prpt: str, device):
        """sample prompt の post-LLM Adapter context を返す (_sample_image_inference と同一の計算)。
        エンコード不能 (cache になく text_encoder もない) なら None。"""
        encoded = None
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            encoded = sample_prompts_te_outputs[prpt]
        elif text_encoder is not None and tokenize_strategy is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
        if encoded is None:
            return None
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded
        if isinstance(prompt_embeds, np.ndarray):
            prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
            attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
            t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
            t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device, dtype=dit.dtype)
        attn_mask = attn_mask.to(device)
        t5_input_ids = t5_input_ids.to(device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(device)
        if getattr(dit, "use_llm_adapter", False):
            ctx = dit.llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            ctx[~t5_attn_mask.bool()] = 0
        else:
            ctx = prompt_embeds
        return ctx

    def on_prompt_start(prompt_dict: dict, accelerator):
        # remember the multiplier in effect prior to this prompt so we can restore it
        saved["multiplier"] = lllite.multiplier

        # multiplier: per-prompt --am (list-form, take first) overrides global --lllite_multiplier
        am = prompt_dict.get("additional_network_multiplier")
        if am is not None and len(am) > 0:
            lllite.set_multiplier(float(am[0]))
        else:
            lllite.set_multiplier(args.lllite_multiplier)

        # control image: per-prompt --cn → controlnet_image
        ci_path = prompt_dict.get("controlnet_image")
        if ci_path is None:
            logger.warning(
                "no control image for sample prompt (use '--cn <path>'); running base DiT without LLLite cond"
            )
            lllite.clear_cond_image()
            return

        if not os.path.isfile(ci_path):
            logger.warning(f"control image not found: {ci_path}; running base DiT without LLLite cond")
            lllite.clear_cond_image()
            return

        # match the dimensions used by _sample_image_inference (rounded to multiple of 16)
        w = prompt_dict.get("width", 512)
        h = prompt_dict.get("height", 512)
        h = max(64, h - h % 16)
        w = max(64, w - w % 16)
        rgb = _load_control_image(ci_path, w, h, accelerator.device, dit_dtype)

        if is_inpaint:
            mk_path = prompt_dict.get("mask_image")
            if mk_path is None:
                logger.warning(
                    "inpaint LLLite: no mask image for sample prompt (use '--mk <path>'); "
                    "running base DiT without LLLite cond"
                )
                lllite.clear_cond_image()
                return
            if not os.path.isfile(mk_path):
                logger.warning(
                    f"inpaint LLLite: mask image not found: {mk_path}; running base DiT without LLLite cond"
                )
                lllite.clear_cond_image()
                return
            mask = _load_mask_image(mk_path, w, h, accelerator.device, dit_dtype)
        else:
            mask = None

        cond_image, cond_mask = build_cond_tensors(
            rgb,
            mask,
            cond_input_space=lllite.cond_input_space,
            cond_in_channels=lllite.cond_in_channels,
            inpaint_masked_input=args.lllite_inpaint_masked_input,
            vae=vae,
        )
        if is_semantic:
            # v3: cond latent を凍結 DiT に通し hidden states を条件源にする。
            # h_ref は t 不変なので prompt 単位で 1 回だけ前計算すればよい
            lllite.clear_cond_image()
            device = cond_image.device
            # uncond context は本体 context 長 (T5 長) へゼロパッドする (長さ 1 のままだと
            # softmax=1 の定数注入で参照特徴が壊れる。build_uncond_ref_context 参照)
            t5_len = tokenize_strategy.t5_max_length if tokenize_strategy is not None else 512

            with torch.no_grad():
                if ref_context == "caption":
                    prompt = prompt_dict.get("prompt", "")
                    ctx_pos = _encode_prompt_context(prompt, device)
                    if ctx_pos is None:
                        logger.warning(
                            "ref_context='caption': cannot encode the sample prompt "
                            "(not cached and no text encoder); falling back to the uncond ref context"
                        )
                        ctx_pos = build_uncond_ref_context(
                            dit, device, cond_image.dtype, pad_to_length=t5_len
                        )
                    # CFG の uncond ブランチ用 (negative prompt の context)。学習の caption dropout
                    # (drop されたサンプルは参照も uncond) と整合する per-branch 切替
                    entries = [(ctx_pos, None)]
                    scale = prompt_dict.get("scale", 7.5)
                    negative_prompt = prompt_dict.get("negative_prompt", "")
                    if scale > 1.0 and negative_prompt is not None:
                        ctx_neg = _encode_prompt_context(negative_prompt, device)
                        if ctx_neg is not None and not torch.equal(ctx_neg, ctx_pos):
                            entries.append((ctx_neg, None))
                    entries = [
                        (
                            ctx,
                            encode_reference_hidden_states(
                                dit, cond_image, lllite.ref_blocks, lllite.ref_timestep, context=ctx
                            ),
                        )
                        for ctx, _ in entries
                    ]
                    if len(entries) == 1:
                        lllite.set_cond_hidden_states(entries[0][1])
                    else:
                        saved["dispatch_handle"] = install_ref_context_dispatch(dit, lllite, entries)
                else:
                    ref_ctx = (
                        build_uncond_ref_context(dit, device, cond_image.dtype, pad_to_length=t5_len)
                        if ref_context == "uncond"
                        else None
                    )
                    h_ref = encode_reference_hidden_states(
                        dit, cond_image, lllite.ref_blocks, lllite.ref_timestep, context=ref_ctx
                    )
                    lllite.set_cond_hidden_states(h_ref)
        else:
            lllite.set_cond_image(cond_image, cond_mask)

    def on_prompt_end(prompt_dict: dict):
        if saved["dispatch_handle"] is not None:
            saved["dispatch_handle"].remove()
            saved["dispatch_handle"] = None
        lllite.clear_cond_image()
        if saved["multiplier"] is not None:
            lllite.set_multiplier(saved["multiplier"])
            saved["multiplier"] = None

    return on_prompt_start, on_prompt_end


def add_anima_lllite_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cond_emb_dim",
        type=int,
        default=32,
        help="conditioning embedding dimension / 条件付け埋め込みの次元数 (default: 32)",
    )
    parser.add_argument(
        "--lllite_mlp_dim",
        type=int,
        default=64,
        help="LLLite MLP (LoRA-rank-like) hidden dim / LLLite の中間次元 (default: 64)",
    )
    parser.add_argument(
        "--lllite_target_layers",
        type=str,
        default="self_attn_q",
        help=(
            "which Linear layers to attach LLLite to. "
            "Either a preset name or a comma-separated list of atomic specifiers. "
            f"presets: {list(LLLITE_PRESETS)}, atomic: {list(LLLITE_ATOMIC_SPECIFIERS)}. "
            "default: self_attn_q"
        ),
    )
    parser.add_argument(
        "--lllite_cond_dim",
        type=int,
        default=64,
        help="conditioning1 trunk channel width / conditioning1 内部の中間チャネル幅 (default: 64)",
    )
    parser.add_argument(
        "--lllite_cond_resblocks",
        type=int,
        default=1,
        help="number of ResBlocks in conditioning1 / conditioning1 の ResBlock 段数 (default: 1)",
    )
    parser.add_argument(
        "--lllite_use_aspp",
        action="store_true",
        help="enable ASPP (Atrous Spatial Pyramid Pooling) at the end of conditioning1 / conditioning1 末尾に ASPP を挿入",
    )
    parser.add_argument(
        "--lllite_dropout",
        type=float,
        default=None,
        help="dropout rate for LLLite mid output / LLLite mid 出力の dropout 率 (default: None)",
    )
    parser.add_argument(
        "--lllite_multiplier",
        type=float,
        default=1.0,
        help="multiplier applied to LLLite output / LLLite 出力に乗算する倍率 (default: 1.0)",
    )
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained LLLite weights to resume from / 学習を再開する LLLite の初期重み",
    )
    parser.add_argument(
        "--lllite_cond_in_channels",
        type=int,
        default=3,
        help=(
            "number of input channels for the LLLite conditioning1 trunk (default: 3, RGB only). "
            "Set to 4 to enable inpainting mode (RGB + 1ch mask). "
            "/ LLLite の conditioning1 入力チャネル数。デフォルト 3 (RGB)、4 で inpainting 用 (RGB+mask)"
        ),
    )
    parser.add_argument(
        "--lllite_cond_input",
        type=str,
        default="pixel",
        choices=list(LLLITE_COND_INPUT_SPACES),
        help=(
            "input space for the control image: 'pixel' feeds the image directly to conditioning1 (default, v2), "
            "'latent' VAE-encodes it first and feeds the 16ch latent (v2.1). "
            "/ 制御画像の入力空間。pixel は画像をそのまま conditioning1 に入力 (既定)、"
            "latent は VAE で encode した 16ch latent を入力する"
        ),
    )
    parser.add_argument(
        "--lllite_inpaint_masked_input",
        action="store_true",
        help=(
            "[inpaint] additionally zero out RGB inside the mask region before concatenating with mask. "
            "Only effective when --lllite_cond_in_channels=4. "
            "/ inpainting 時、RGB の mask 域を 0 で穴埋めしてから concat する (cond_in_channels=4 のときのみ有効)"
        ),
    )
    parser.add_argument(
        "--lllite_trunk",
        type=str,
        default="stem",
        choices=list(LLLITE_TRUNK_TYPES),
        help=(
            "conditioning trunk: 'stem' feeds the cond input to a scratch conv stem (v2/v2.1, default), "
            "'semantic' feeds the frozen DiT's hidden states of the cond latent to the trunk (v3). "
            "'semantic' requires --lllite_cond_input latent and does not support inpainting. "
            "/ 条件付けトランク。stem は conv stem (v2/v2.1)、semantic は凍結 DiT の中間 hidden states を"
            "条件源にする (v3、--lllite_cond_input latent 必須・inpaint 非対応)"
        ),
    )
    parser.add_argument(
        "--lllite_ref_block",
        type=str,
        default=None,
        help=(
            "[semantic] DiT block index(es) whose output hidden states feed the trunk "
            "(default: num_blocks // 2). Comma-separated indices (e.g. '2,13') enable the dual/multi "
            "concat trunk: one reference forward runs blocks up to the max index and the hidden states "
            "of each listed block are concatenated (shallower blocks cost nothing extra). "
            "Index -1 selects the x_embedder output (the embedded cond latent itself, before any "
            "block) as a pure-appearance source; note argparse needs the '=' form for a leading "
            "minus (--lllite_ref_block=-1,14). "
            "/ semantic trunk が hidden states を取り出すブロック index (既定: num_blocks // 2)。"
            "'2,13' のようにカンマ区切りで複数指定すると dual/multi concat になる "
            "(参照フォワードは最大 index まで 1 回だけ実行、浅いブロックの追加コストはゼロ)。"
            "-1 は x_embedder 出力 (latent の埋め込みそのもの) を純外観の条件源として使う "
            "(先頭がマイナスの値は --lllite_ref_block=-1,14 の = 形式で指定)"
        ),
    )
    parser.add_argument(
        "--lllite_ref_timestep",
        type=float,
        default=0.0,
        help=(
            "[semantic] timestep for the reference forward over the cond latent, in the [0, 1] scale "
            "(0 = clean, default). / 参照フォワードの timestep ([0,1] スケール、0 = clean)"
        ),
    )
    parser.add_argument(
        "--lllite_ref_context",
        type=str,
        default="zero",
        choices=list(LLLITE_REF_CONTEXT_MODES),
        help=(
            "[semantic] text context for the reference forward. 'zero' (default): zero context, "
            "fully text-independent. 'uncond': a constant empty-prompt context (in-distribution, "
            "still text-independent and cacheable; recommended). 'caption': the same context as the "
            "main forward (instruct-pix2pix style; the frozen DiT computes the prompt-vs-image "
            "mismatch into the reference features; caption dropout stays consistent automatically). "
            "/ 参照フォワードの context。zero=ゼロ context (既定)、uncond=空プロンプト定数 (分布内・"
            "テキスト非依存のまま、推奨)、caption=本体と同じ context (instruct-pix2pix 型)"
        ),
    )
    parser.add_argument(
        "--lllite_gate",
        type=str,
        default="scalar",
        choices=list(LLLITE_GATE_MODES),
        help=(
            "[semantic] gate mode: 'scalar' = per-token scalar gate on the value path (v3 default), "
            "'none' = no gate; the value path alone learns both where and what to inject "
            "(ablation for cases where a per-token scalar cannot express keep-geometry / "
            "release-appearance at the same token). Ignored for trunk='stem'. "
            "/ semantic trunk のゲート。scalar=per-token スカラーゲート (v3 既定)、"
            "none=ゲートなし (値パス単独で注入の場所と内容を学ぶ ablation)。stem では無視"
        ),
    )
    # --conditioning_data_dir は args_util.add_dataset_arguments 側で既に定義済み


def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    flux_train_utils.log_timestep_sampling_info(args)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True

    # MVP では未対応の機能を明示的に弾く
    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ), "blocks_to_swap is not supported in Anima ControlNet-LLLite training (MVP)"
    assert not args.cpu_offload_checkpointing, (
        "cpu_offload_checkpointing is not supported in Anima ControlNet-LLLite training (MVP)"
    )
    assert not args.unsloth_offload_checkpointing, (
        "unsloth_offload_checkpointing is not supported in Anima ControlNet-LLLite training (MVP)"
    )
    assert not args.deepspeed, "deepspeed is not supported in Anima ControlNet-LLLite training (MVP)"
    assert not args.fused_backward_pass, (
        "fused_backward_pass is not supported in Anima ControlNet-LLLite training (MVP)"
    )

    # per-block torch.compile の排他チェック (anima_train_network.assert_extra_args と同等)
    if args.compile:
        assert not args.torch_compile, (
            "--compile (per-block torch.compile) and --torch_compile (accelerate dynamo) cannot be used together"
            " / --compile（ブロック単位torch.compile）と--torch_compile（accelerate dynamo）は併用できません"
        )
        assert not (args.compile_fullgraph and args.split_attn), (
            "--compile_fullgraph cannot be used with --split_attn (split attention uses dynamic control flow)"
            " / --compile_fullgraphは--split_attnと併用できません（split attentionは動的な制御フローを使用します）"
        )

    cache_latents = args.cache_latents

    # cond 入力空間 (v2.1). latent モードでは cond 画像を毎ステップ VAE encode するため VAE を常駐させる
    is_latent_cond = args.lllite_cond_input == "latent"

    if args.seed is not None:
        set_seed(args.seed)

    # latents caching strategy
    if cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # dataset (ControlNet 形式)
    if args.dataset_class is not None:
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)
        val_dataset_group = None
    else:
        # ControlNet 用 sanitizer: dreambooth=False, finetuning=False, controlnet=True, dropout=True
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, False, True, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "conditioning_data_dir"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning("ignore following options because config file is found: {0}".format(", ".join(ignored)))
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
                            args.train_data_dir,
                            args.conditioning_data_dir,
                            args.caption_extension,
                        )
                    }
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)  # Qwen-Image VAE /8 * patch /2

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, False, False
                )
            )
        logger.info("Loading tokenizers...")
        weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)
        qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen3_tokenizer,
            t5_tokenizer=t5_tokenizer,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        train_dataset_group.set_current_strategies()
        dataset_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify train_data_dir / conditioning_data_dir / dataset_config.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, color_aug/random_crop cannot be used"
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), "when caching text encoder output, shuffle_caption / token_warmup_step / caption_tag_dropout_rate cannot be used"

    # accelerator
    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    # tokenizers and strategies
    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        qwen3_text_encoder.to(accelerator.device)
        qwen3_text_encoder.eval()

        text_encoder_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, is_partial=False
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

        if args.sample_prompts is not None:
            logger.info(f"Cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
            prompts = sampling.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, [qwen3_text_encoder], tokens_and_masks
                            )

        accelerator.wait_for_everyone()

        qwen3_text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # VAE
    logger.info("Loading Anima VAE...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)

    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        train_dataset_group.new_cache_latents(vae, accelerator)
        if is_latent_cond:
            # latent cond モードでは学習ループ・サンプル生成で毎回 cond 画像を encode するため GPU に残す
            logger.info("keeping VAE on GPU for latent cond encoding (--lllite_cond_input latent)")
        else:
            vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # DiT (frozen)
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(
        "cpu", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=None
    )

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(
            cpu_offload=args.cpu_offload_checkpointing,
            unsloth_offload=args.unsloth_offload_checkpointing,
        )

    dit.requires_grad_(False)

    # inpainting (4ch) フラグの早期検証
    if args.lllite_cond_in_channels < 1:
        raise ValueError(f"--lllite_cond_in_channels must be >= 1, got {args.lllite_cond_in_channels}")
    if is_latent_cond and args.lllite_cond_in_channels not in (3, 4):
        raise ValueError(
            "--lllite_cond_input latent supports --lllite_cond_in_channels 3 (RGB) or 4 (RGB+mask), "
            f"got {args.lllite_cond_in_channels}"
        )
    if args.lllite_inpaint_masked_input and args.lllite_cond_in_channels != 4:
        logger.warning(
            f"--lllite_inpaint_masked_input is only effective when --lllite_cond_in_channels=4 "
            f"(got {args.lllite_cond_in_channels}); flag will be ignored at runtime"
        )
    is_inpaint = args.lllite_cond_in_channels == 4

    # v3 (semantic trunk) の制約検証
    is_semantic_trunk = args.lllite_trunk == "semantic"
    if is_semantic_trunk:
        # ref_block 指定のパースエラーはここで早期に落とす (構築時より原因が分かりやすい)
        lllite_module.parse_ref_blocks(args.lllite_ref_block)
        if not is_latent_cond:
            raise ValueError("--lllite_trunk semantic requires --lllite_cond_input latent")
        if args.lllite_cond_in_channels != 3:
            raise ValueError(
                "--lllite_trunk semantic does not support inpainting (--lllite_cond_in_channels 4) yet"
            )
        if args.cond_emb_dim < 64:
            logger.warning(
                f"--cond_emb_dim {args.cond_emb_dim} is small for the semantic trunk; the DiT hidden "
                f"states carry semantic features, so 128 or so is recommended to avoid a bottleneck"
            )
    elif args.lllite_ref_context != "zero":
        raise ValueError("--lllite_ref_context is only supported with --lllite_trunk semantic")

    # Build LLLite (DiT を走査して各 Attention Linear に貼る)
    logger.info("Building ControlNet-LLLite (Anima)...")
    lllite = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.lllite_mlp_dim,
        target_layers=args.lllite_target_layers,
        dropout=args.lllite_dropout,
        multiplier=args.lllite_multiplier,
        cond_dim=args.lllite_cond_dim,
        cond_resblocks=args.lllite_cond_resblocks,
        use_aspp=args.lllite_use_aspp,
        cond_in_channels=args.lllite_cond_in_channels,
        inpaint_masked_input=args.lllite_inpaint_masked_input,
        cond_input_space=args.lllite_cond_input,
        trunk=args.lllite_trunk,
        ref_block=args.lllite_ref_block,
        ref_timestep=args.lllite_ref_timestep,
        ref_context=args.lllite_ref_context,
        gate=args.lllite_gate,
    )

    if args.network_weights is not None:
        # metadata の target_layers と一致しているかは load 時に warning のみ (strict=False)
        load_lllite_weights(lllite, args.network_weights, strict=False)

    lllite.apply_to()

    wrapper = AnimaControlNetLLLiteWrapper(dit, lllite)

    # Optimizer
    trainable_params = list(lllite.parameters())
    n_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    accelerator.print(f"number of LLLite modules: {len(lllite.lllite_modules)}")
    accelerator.print(f"number of trainable parameters: {n_trainable:,}")

    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = optimizer_util.get_optimizer(args, trainable_params=trainable_params)
    optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer, args)

    # dataloader
    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # dtype: DiT は凍結だが forward 通過時の autocast を使うので weight_dtype に揃える
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed_precision='fp16'"
        accelerator.print("enable full fp16 training.")
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
    else:
        # LLLite 自体は fp32 で学習、DiT は weight_dtype
        dit_weight_dtype = weight_dtype
    dit.to(dit_weight_dtype)
    dit.to(accelerator.device)

    # LLLite は fp32 (full_*16 のときは weight_dtype)
    lllite_dtype = torch.float32
    if args.full_fp16 or args.full_bf16:
        lllite_dtype = weight_dtype
    lllite.to(lllite_dtype)
    lllite.to(accelerator.device)

    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)
    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    clean_memory_on_device(accelerator.device)

    # accelerator.prepare — wrapper を渡す
    wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wrapper, optimizer, train_dataloader, lr_scheduler
    )

    if args.full_fp16:
        accelerator_setup.patch_accelerator_for_fp16_training(accelerator)

    # CUDA perf switches are independent of torch.compile; apply whenever requested.
    compile_utils.apply_cuda_optimizations(args)

    if args.compile:
        # per-block torch.compile を凍結 DiT のブロックに適用する。LLLite の forward 差し替え
        # (apply_to) と accelerator.prepare の後でなければならない。block swap は MVP で無効
        # のため disable_linear=False 固定。LLLite モジュールは対象 Linear の forward を差し替え
        # ているため、compile は patch 済みの forward を取り込む (cond_emb はガード付き入力扱い)。
        dit_to_compile = accelerator.unwrap_model(wrapper).dit
        compile_utils.compile_transformer(args, dit_to_compile, [dit_to_compile.blocks], disable_linear=False)

    args_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch計算
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training (Anima ControlNet-LLLite)")
    accelerator.print(f"  num train images x repeats: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "anima_controlnet_lllite" if args.log_tracker_name is None else args.log_tracker_name,
            config=args_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # sample image hooks: inject control image / multiplier into LLLite around each prompt
    on_prompt_start, on_prompt_end = _make_lllite_sample_hooks(
        args,
        accelerator.unwrap_model(wrapper).lllite,
        dit_weight_dtype,
        vae,
        dit=accelerator.unwrap_model(wrapper).dit,
        tokenize_strategy=tokenize_strategy,
        text_encoding_strategy=text_encoding_strategy,
        text_encoder=qwen3_text_encoder,
        sample_prompts_te_outputs=sample_prompts_te_outputs,
    )

    def _sample_images(epoch_arg, step_arg):
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch_arg,
            step_arg,
            accelerator.unwrap_model(wrapper).dit,
            vae,
            qwen3_text_encoder,
            tokenize_strategy,
            text_encoding_strategy,
            sample_prompts_te_outputs,
            on_prompt_start=on_prompt_start,
            on_prompt_end=on_prompt_end,
        )

    # --sample_at_first
    optimizer_eval_fn()
    _sample_images(0, global_step)
    optimizer_train_fn()

    # save helper (LLLite のみ)
    def _save_lllite(ckpt_file: str):
        sai_metadata = model_io.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        sai_metadata["modelspec.architecture"] = "anima-preview/control-net-lllite"
        sai_metadata["lllite.version"] = (
            LLLITE_ARCH_VERSION_SEMANTIC if args.lllite_trunk == "semantic" else LLLITE_ARCH_VERSION
        )
        sai_metadata["lllite.cond_emb_dim"] = str(args.cond_emb_dim)
        sai_metadata["lllite.mlp_dim"] = str(args.lllite_mlp_dim)
        sai_metadata["lllite.target_layers"] = args.lllite_target_layers
        unwrapped = accelerator.unwrap_model(wrapper).lllite
        # canonical atomic 形式も記録 (推論時の解決と log 用、preset 名と冗長だが互換維持)
        sai_metadata["lllite.target_atomics"] = unwrapped.target_atomics_str
        sai_metadata["lllite.cond_dim"] = str(args.lllite_cond_dim)
        sai_metadata["lllite.cond_resblocks"] = str(args.lllite_cond_resblocks)
        sai_metadata["lllite.use_aspp"] = "true" if args.lllite_use_aspp else "false"
        if args.lllite_use_aspp:
            sai_metadata["lllite.aspp_dilations"] = ",".join(str(d) for d in unwrapped.aspp_dilations)
        sai_metadata["lllite.cond_input_space"] = args.lllite_cond_input
        sai_metadata["lllite.cond_in_channels"] = str(args.lllite_cond_in_channels)
        sai_metadata["lllite.inpaint_masked_input"] = (
            "true" if args.lllite_inpaint_masked_input else "false"
        )
        sai_metadata["lllite.trunk"] = args.lllite_trunk
        if args.lllite_trunk == "semantic":
            sai_metadata["lllite.ref_block"] = unwrapped.ref_blocks_str
            sai_metadata["lllite.ref_timestep"] = str(unwrapped.ref_timestep)
            sai_metadata["lllite.ref_context"] = unwrapped.ref_context
            sai_metadata["lllite.gate"] = unwrapped.gate_mode
        save_lllite_model(ckpt_file, unwrapped, dtype=save_dtype, metadata=sai_metadata)

    def _save_step(global_step_: int, epoch_: int):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, global_step_)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_lllite(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_stepwise(args, accelerator, global_step_)
        remove_step_no = checkpoint_io.get_remove_step_no(args, global_step_)
        if remove_step_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    def _save_epoch(epoch_no: int):
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_lllite(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        remove_epoch_no = checkpoint_io.get_remove_epoch_no(args, epoch_no)
        if remove_epoch_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    loss_recorder = logging_util.LossRecorder()
    epoch = 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        wrapper.train()
        # DiT は凍結だが gradient_checkpointing 有効時に train モードが必要
        accelerator.unwrap_model(wrapper).dit.train() if args.gradient_checkpointing else accelerator.unwrap_model(wrapper).dit.eval()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(wrapper):
                # latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)
                else:
                    with torch.no_grad():
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # text encoder outputs
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    caption_dropout_rates = text_encoder_outputs_list[-1]
                    text_encoder_outputs_list = text_encoder_outputs_list[:-1]
                    text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(
                        *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
                    )
                    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_outputs_list
                else:
                    input_ids_list = batch["input_ids_list"]
                    with torch.no_grad():
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], input_ids_list
                        )

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)

                # noise + timesteps
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents, noise, accelerator.device, dit_weight_dtype
                )
                timesteps = timesteps / 1000.0
                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                # padding mask
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device)

                # cond image: dataset 側で IMAGE_TRANSFORMS により [-1,1] 正規化済み
                cond_rgb = batch["conditioning_images"].to(accelerator.device, dtype=dit_weight_dtype)

                # inpainting: ランダム mask をバッチ毎に生成する
                if is_inpaint:
                    bs_c, _, h_c, w_c = cond_rgb.shape
                    mask = _generate_random_masks_for_batch(
                        bs_c, h_c, w_c, accelerator.device, dit_weight_dtype
                    )
                else:
                    mask = None

                # pixel モード: (B, 3 or 4, H, W) / latent モード: VAE encode 済み (B,16,H/8,W/8)
                # + pixel 解像度の mask を別テンソルで渡す。VAE encode は autocast の外 (no_grad)。
                cond_image, cond_mask = build_cond_tensors(
                    cond_rgb,
                    mask,
                    cond_input_space=args.lllite_cond_input,
                    cond_in_channels=args.lllite_cond_in_channels,
                    inpaint_masked_input=args.lllite_inpaint_masked_input,
                    vae=vae,
                )

                # 5D化
                noisy_model_input = noisy_model_input.unsqueeze(2)  # (B, C, 1, H, W)

                with accelerator.autocast():
                    model_pred = wrapper(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        cond_image=cond_image,
                        cond_mask=cond_mask,
                        padding_mask=padding_mask,
                        source_attention_mask=attn_mask,
                        t5_input_ids=t5_input_ids,
                        t5_attn_mask=t5_attn_mask,
                    )
                model_pred = model_pred.squeeze(2)

                target = noise - latents

                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                if weighting is not None:
                    loss = loss * weighting

                loss_weights = batch["loss_weights"]
                loss = loss * loss_weights
                loss = loss.mean()

                try:
                    accelerator.backward(loss)
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"OOM at step={global_step} epoch={epoch} "
                        f"latents={tuple(latents.shape)} "
                        f"prompt_embeds={tuple(prompt_embeds.shape)} "
                        f"cond_image={tuple(cond_image.shape)}"
                    )
                    try:
                        logger.error(torch.cuda.memory_summary(abbreviated=False))
                    except Exception as e:
                        logger.error(f"failed to dump memory_summary: {e}")
                    raise

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = list(accelerator.unwrap_model(wrapper).lllite.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # cond_emb の参照を残さない (次 step で上書きされるが、保険)
            accelerator.unwrap_model(wrapper).lllite.clear_cond_image()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                optimizer_eval_fn()
                _sample_images(None, global_step)
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    _save_step(global_step, epoch)
                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            progress_bar.set_postfix(**{"avr_loss": avr_loss})

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average, "epoch": epoch + 1}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
            _save_epoch(epoch + 1)
        _sample_images(epoch + 1, global_step)
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        checkpoint_io.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = checkpoint_io.get_last_ckpt_name(args, "." + args.save_model_as)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving final checkpoint: {ckpt_file}")
        _save_lllite(ckpt_file)
        logger.info("model saved.")

    del accelerator


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    args_util.add_sd_models_arguments(parser)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_training_arguments(parser, False)
    args_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    args_util.add_sd_saving_arguments(parser)
    args_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload gradient checkpointing to CPU",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload activations to CPU async",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead",
    )

    add_anima_lllite_arguments(parser)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        train(args)
