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
