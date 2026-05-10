"""Command-line argument definitions and configuration helpers.

Hosts every ``add_*_arguments`` parser registration along with the verification
and configuration utilities (``verify_*``, ``get_sanitized_config_or_none``,
``read_config_from_file``, ``resume_from_local_or_hf_if_specified``).
Extracted from ``library.train_util`` and re-exported there for backward
compatibility.
"""

import argparse
import asyncio
import os
import pathlib

import toml
from huggingface_hub import hf_hub_download

import library.huggingface_util as huggingface_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, "
        "Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit'.",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )

    # parser.add_argument(
    #     "--optimizer_schedulefree_wrapper",
    #     action="store_true",
    #     help="use schedulefree_wrapper any optimizer / 任意のオプティマイザにschedulefree_wrapperを使用",
    # )

    # parser.add_argument(
    #     "--schedulefree_wrapper_args",
    #     type=str,
    #     default=None,
    #     nargs="*",
    #     help='additional arguments for schedulefree_wrapper (like "momentum=0.9 weight_decay_at_y=0.1 ...") / オプティマイザの追加引数（例： "momentum=0.9 weight_decay_at_y=0.1 ..."）',
    # )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage. Only available in SDXL, SD3 and FLUX"
        " / バックワードパスとオプティマイザステップを組み合わせてVRAMの使用量を削減します。SDXL、SD3、FLUXでのみ利用可能",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr と warmup decay スケジューラ で有効",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名"
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving / 保存時に精度を変更して保存する",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end / optimizerなど学習状態も含めたstateを学習完了時に保存する",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う",
    )
    parser.add_argument(
        "--torch_compile", action="store_true", help="use torch.compile (requires PyTorch 2.0) / torch.compile を使う"
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        # available backends:
        # https://github.com/huggingface/accelerate/blob/d1abd59114ada8ba673e1214218cb2878c13b82d/src/accelerate/utils/dataclasses.py#L376-L388C5
        # https://pytorch.org/docs/stable/torch.compiler.html
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "tensort",
            "ipex",
            "tvm",
        ],
        help="dynamo backend type (default is inductor) / dynamoのbackendの種類（デフォルトは inductor）",
    )
    parser.add_argument("--xformers", action="store_true", help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument(
        "--full_fp16",
        action="store_true",
        help="fp16 training including gradients, some models are not supported / 勾配も含めてfp16で学習する、一部のモデルではサポートされていません",
    )
    parser.add_argument(
        "--full_bf16",
        action="store_true",
        help="bf16 training including gradients, some models are not supported / 勾配も含めてbf16で学習する、一部のモデルではサポートされていません",
    )  # TODO move to SDXL training, because it is not supported by SD1/2
    parser.add_argument(
        "--fp8_base",
        action="store_true",
        help="use fp8 for base model, some models are not supported / base modelにfp8を使う、一部のモデルではサポートされていません",
    )

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset. / noise offsetにおいて、0からnoise_offsetの間でランダムな強度を使用します。",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. used for regularization. recommended value: around 0.1 (from arxiv.org/abs/2301.11706) "
        + "/  input perturbation noiseを有効にする。正則化に使用される。推奨値: 0.1程度 (arxiv.org/abs/2301.11706 より)",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise."
        + "/ input perturbation noiseにおいて、0からip_noise_gammaの間でランダムな強度を使用します。",
    )
    # parser.add_argument(
    #     "--perlin_noise",
    #     type=int,
    #     default=None,
    #     help="enable perlin noise and set the octaves / perlin noiseを有効にしてoctavesをこの値に設定する",
    # )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR / noise schedulerのbetasを修正して、zero terminal SNRを強制する",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0) / U-Net学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000) / U-Net学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber", "smooth_l1"],
        help="The type of loss function to use (L1, L2, Huber, or smooth L1), default is L2 / 使用する損失関数の種類（L1、L2、Huber、またはsmooth L1）、デフォルトはL2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss (constant, exponential, or SNR-based). Only used when loss_type is 'huber' or 'smooth_l1'. default is snr"
        + " / Huber損失のスケジューリング方法（constant、exponential、またはSNRベース）。loss_typeが'huber'または'smooth_l1'の場合に有効、デフォルトは snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The Huber loss decay parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 0.1"
        " / Huber損失の減衰パラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは0.1",
    )

    parser.add_argument(
        "--huber_scale",
        type=float,
        default=1.0,
        help="The Huber loss scale parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. default is 1.0"
        " / Huber損失のスケールパラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは1.0",
    )

    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込む等（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）",
    )
    parser.add_argument(
        "--highvram",
        action="store_true",
        help="disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM) "
        + "/ VRAMが少ない環境向け最適化を無効にする。たとえば各latentのキャッシュ後のCUDAキャッシュクリアを行わない等（VRAMが多い環境向け）",
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する"
    )
    if support_dreambooth:
        # DreamBooth training
        parser.add_argument(
            "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み"
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--masked_loss",
        action="store_true",
        help="apply mask for calculating loss. conditioning_data_dir is required for dataset. / 損失計算時にマスクを適用する。datasetにはconditioning_data_dirが必要",
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    # Text encoder related arguments
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)"
        + " / text encoderのバッチサイズ（デフォルト: None, データセットのバッチサイズを使用）",
    )

    # Model loading optimization
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )

    # Training arguments. partial copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"],
        help="weighting scheme for timestep distribution. Default is uniform, uniform and none are the same behavior"
        " / タイムステップ分布の重み付けスキーム、デフォルトはuniform、uniform と none は同じ挙動",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )

    # offloading
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] "
        "Sets the number of blocks to swap during the forward and backward passes."
        "Increasing this number lowers the overall VRAM used during training at the expense of training speed (s/it)."
        " / 順伝播および逆伝播中にスワップするブロックの数を設定します。"
        "この数を増やすと、トレーニング中のVRAM使用量が減りますが、トレーニング速度（s/it）も低下します。",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


# verify command line args for training
def verify_command_line_training_args(args: argparse.Namespace):
    # if wandb is enabled, the command line is exposed to the public
    # check whether sensitive options are included in the command line arguments
    # if so, warn or inform the user to move them to the configuration file
    # wandbが有効な場合、コマンドラインが公開される
    # 学習用のコマンドライン引数に敏感なオプションが含まれているかどうかを確認し、
    # 含まれている場合は設定ファイルに移動するようにユーザーに警告または通知する

    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"  # "all" or "wandb"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれています。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
            )

    # if path is absolute, it may include sensitive information
    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file or use relative path."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれており、絶対パスです。コマンドラインは公開されるため、`.toml`ファイルに移動するか、相対パスを使用することをお勧めします。"
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            f"wandb is enabled, but option `config_file` is included in the command line. Because the command line is exposed to the public, please be careful about the information included in the path."
            + f" / wandbが有効で、かつオプション `config_file` がコマンドラインに含まれています。コマンドラインは公開されるため、パスに含まれる情報にご注意ください。"
        )

    # other sensitive options
    if args.huggingface_repo_id is not None and args.huggingface_repo_visibility != "public":
        logger.info(
            f"wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
            + f" / wandbが有効で、かつオプション huggingface_repo_id がコマンドラインに含まれており、huggingface_repo_visibility が 'public' ではありません。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
        )


def verify_training_args(args: argparse.Namespace):
    r"""
    Verify training arguments. Also reflect highvram option to global variable
    学習用引数を検証する。あわせて highvram オプションの指定をグローバル変数に反映する
    """
    from library.accelerator_setup import enable_high_vram

    enable_high_vram(args)

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled / cache_latents_to_diskが有効なため、cache_latentsを有効にします"
        )

    if getattr(args, "train_inpainting", False) and getattr(args, "cache_latents", False):
        raise ValueError(
            "train_inpainting and cache_latents cannot be used together. "
            "Inpainting masks are generated randomly per step from the original image, "
            "so the image must be read on every step. "
            "Disable cache_latents (and cache_latents_to_disk) when using --train_inpainting."
        )

    # noise_offset, perlin_noise, multires_noise_iterations cannot be enabled at the same time
    # # Listを使って数えてもいいけど並べてしまえ
    # if args.noise_offset is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "noise_offset and multires_noise_iterations cannot be enabled at the same time / noise_offsetとmultires_noise_iterationsを同時に有効にできません"
    #     )
    # if args.noise_offset is not None and args.perlin_noise is not None:
    #     raise ValueError("noise_offset and perlin_noise cannot be enabled at the same time / noise_offsetとperlin_noiseは同時に有効にできません")
    # if args.perlin_noise is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "perlin_noise and multires_noise_iterations cannot be enabled at the same time / perlin_noiseとmultires_noise_iterationsを同時に有効にできません"
    #     )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset / adaptive_noise_scaleを使用するにはnoise_offsetが必要です")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization / scale_v_pred_loss_like_noise_predはv_parameterizationが有効なときのみ有効にできます"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError(
            "v_pred_like_loss cannot be enabled with v_parameterization / v_pred_like_lossはv_parameterizationが有効なときには有効にできません"
        )

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            f"zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
            + " / zero_terminal_snrが有効ですが、v_parameterizationが有効ではありません。学習結果は想定外になる可能性があります"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled / sample_every_n_epochsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled / sample_every_n_stepsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    # dataset common
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ"
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information (caption and image size) for faster dataset loading. only available for DreamBooth"
        + " / メタ情報（キャプションとサイズ）をキャッシュしてデータセット読み込みを高速化する。DreamBooth方式のみ有効",
    )
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle separated caption / 区切られたcaptionの各要素をshuffleする"
    )
    parser.add_argument("--caption_separator", type=str, default=",", help="separator for caption / captionの区切り文字")
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子"
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts. Tokens before this separator will not be shuffled. If not specified, '--keep_tokens' will be used to determine the fixed number of tokens."
        + " / captionを固定部分と可変部分に分けるためのカスタム区切り文字。この区切り文字より前のトークンはシャッフルされない。指定しない場合、'--keep_tokens'が固定部分のトークン数として使用される。",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption. This separator is replaced to caption_separator after dropping/shuffling caption"
        + " / captionのセカンダリ区切り文字。この区切り文字はcaptionのドロップやシャッフル後にcaption_separatorに置き換えられる",
    )
    parser.add_argument(
        "--enable_wildcard",
        action="store_true",
        help="enable wildcard for caption (e.g. '{image|picture|rendition}') / captionのワイルドカードを有効にする（例：'{image|picture|rendition}'）",
    )
    parser.add_argument(
        "--caption_prefix",
        type=str,
        default=None,
        help="prefix for caption text / captionのテキストの先頭に付ける文字列",
    )
    parser.add_argument(
        "--caption_suffix",
        type=str,
        default=None,
        help="suffix for caption text / captionのテキストの末尾に付ける文字列",
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）",
    )
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents / latentのcache時のバッチサイズ"
    )
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）",
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache (latent and text encoder output). Cache file existence check is always performed, and cache processing is performed if the file does not exist"
        " / cacheの内容の検証をスキップする（latentとテキストエンコーダの出力）。キャッシュファイルの存在確認は常に行われ、ファイルがなければキャッシュ処理が行われる",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="minimum resolution for buckets, must be divisible by bucket_reso_steps "
        " / bucketの最小解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="maximum resolution for buckets, must be divisible by bucket_reso_steps "
        " / bucketの最大解像度、bucket_reso_stepsで割り切れる必要があります",
    )
    parser.add_argument(
        "--skip_image_resolution",
        type=str,
        default=None,
        help="images not larger than this resolution will be skipped ('size' or 'width,height')"
        " / この解像度以下の画像はスキップされます（'サイズ'指定、または'幅,高さ'指定）",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します",
    )
    parser.add_argument(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"],
        help="Resize interpolation when required. Default: area Options: lanczos, nearest, bilinear, bicubic, area / 必要に応じてサイズ補間を変更します。デフォルト: area オプション: lanczos, nearest, bilinear, bicubic, area",
    )
    parser.add_argument(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する",
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）",
    )
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="use alpha channel as mask for training / 画像のアルファチャンネルをlossのマスクに使用する",
    )

    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class) / 任意のデータセットを用いるときのクラス名 (package.module.Class)",
    )

    parser.add_argument(
        "--train_inpainting",
        action="store_true",
        help="train an inpainting model / インペイントモデルを学習する",
    )

    if support_caption_dropout:
        # Textual Inversion はcaptionのdropoutをsupportしない
        # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
        parser.add_argument(
            "--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合"
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",
        )

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument(
            "--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ"
        )

    if support_caption:
        # caption dataset
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            logger.error(f"Config file already exists. Aborting... / 出力先の設定ファイルが既に存在します: {config_path}")
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info(f"Saved config file / 設定ファイルを保存しました: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError(
            "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
        )
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)
