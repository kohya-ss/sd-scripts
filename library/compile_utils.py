"""torch.compile helpers for DiT-based models.

Ported / adapted from Musubi Tuner (PR kohya-ss/musubi-tuner#722).

The key design choice is **per-block compilation**: instead of compiling the whole
transformer at once, each transformer block (which all share the same structure) is
compiled individually. This keeps the dynamo cache small (one compiled artifact reused
across blocks), avoids recompilation blow-up, and coexists with block swapping (CPU<->GPU
offloading) because swapped blocks can opt out of compilation per Linear layer.

Currently wired up for Anima only. The helpers are model-agnostic, so other DiT trainers
can reuse them by passing their own list of block ModuleLists as ``target_blocks``.
"""

import argparse
from typing import Union

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_compile_arguments(parser: argparse.ArgumentParser):
    """Register the per-block torch.compile (``--compile*``) and CUDA performance (``--cuda_*``) arguments.

    Shared by the DiT trainers which use ``compile_transformer`` / ``apply_cuda_optimizations``. ``--compile`` is
    independent of the legacy accelerate-based ``--torch_compile``; the two cannot be used together.
    """
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable per-block torch.compile for the DiT (requires Triton). Cannot be used with --torch_compile."
        + " / DiTのブロック単位でtorch.compileを有効にする（Tritonが必要）。--torch_compileとは併用できません。",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor) / torch.compileのバックエンド（デフォルト: inductor）",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: default, recommended for training) / torch.compileのモード（デフォルト: default、学習に推奨）",
    )
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default=None,
        choices=["true", "false", "auto"],
        help="Dynamic shapes mode for torch.compile (default: None, same as auto). On Windows, 'true' requires the Visual Studio 2022 C++ compiler."
        + " / torch.compileの動的形状モード（デフォルト: None、autoと同じ）。Windowsで'true'を使うにはVisual Studio 2022のC++コンパイラが必要です。",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile. Cannot be used with --split_attn."
        + " / torch.compileでフルグラフモードを有効にする。--split_attnとは併用できません。",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        type=int,
        default=None,
        help="Set torch._dynamo.config.cache_size_limit (default: PyTorch default, typically 8-32, recommended: 32)"
        + " / torch._dynamo.config.cache_size_limitを設定（デフォルト: PyTorchのデフォルト、通常8-32、推奨: 32）",
    )
    parser.add_argument(
        "--cuda_allow_tf32",
        action="store_true",
        help="Allow TF32 precision on Ampere or newer GPUs (improves performance) / Ampere以降のGPUでTF32を許可する（パフォーマンス向上）",
    )
    parser.add_argument(
        "--cuda_cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark mode (may improve performance) / cuDNNのベンチマークモードを有効にする（パフォーマンスが向上する可能性がある）",
    )


def disable_linear_from_compile(module: torch.nn.Module):
    """Disable torch.compile for every Linear-like submodule (class name ending with 'Linear').

    Used for blocks that are swapped between CPU and GPU: their weights move across devices
    each step, which conflicts with a compiled graph. We replace ``forward`` with a
    ``torch._dynamo.disable()``-wrapped eager version so dynamo treats it as a graph break.
    """
    for sub_module in module.modules():
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def apply_cuda_optimizations(args: argparse.Namespace):
    """Apply optional CUDA performance switches (TF32 / cuDNN benchmark) based on args."""
    if getattr(args, "cuda_allow_tf32", False):
        logger.info("Enabling TF32 for matmul and cuDNN (Ampere or newer GPUs)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(args, "cuda_cudnn_benchmark", False):
        logger.info("Enabling cuDNN benchmark mode")
        torch.backends.cudnn.benchmark = True


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[Union[torch.nn.ModuleList, list[torch.nn.Module]]],
    disable_linear: bool,
) -> torch.nn.Module:
    """Compile each block in ``target_blocks`` individually with torch.compile.

    Args:
        args: parsed arguments providing ``compile_backend`` / ``compile_mode`` /
            ``compile_dynamic`` / ``compile_fullgraph`` / ``compile_cache_size_limit``.
        transformer: the model owning the blocks (returned as-is for convenience).
        target_blocks: list of ModuleLists (or plain lists) whose entries are compiled
            in place; ``blocks[i]`` is replaced by its compiled version.
        disable_linear: when True, disable compilation for Linear layers in the given
            blocks first (required for swapped blocks under block swapping).
    """
    if disable_linear:
        logger.info("Disabling Linear layers from torch.compile for block-swapped blocks...")
        for blocks in target_blocks:
            for block in blocks:
                disable_linear_from_compile(block)

    compile_dynamic = None
    if args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

    logger.info(
        f"Compiling DiT blocks with torch.compile: backend={args.compile_backend}, mode={args.compile_mode}, "
        f"dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )

    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit

    # nn.Module の tensor 属性 (例: ControlNet-LLLite が注入する self.cond_emb) は、
    # 既定では dynamic=True でも shape が static に specialize され、解像度バケット毎に
    # recompile を誘発する。forward 入力ではなく属性経由で渡る可変長テンソルを dynamic
    # 対象に含めるため、この強制 static 化を無効化する。Parameter (学習重み) は形状不変
    # なので force_parameter_static_shapes は既定 (True) のままにしておく。
    if hasattr(torch._dynamo.config, "force_nn_module_property_static_shapes"):
        torch._dynamo.config.force_nn_module_property_static_shapes = False

    for blocks in target_blocks:
        for i, block in enumerate(blocks):
            blocks[i] = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
    return transformer
