"""Generate sample images of new prompts in a training which is running with `--run_control`, even if the text encoder
is not in the training process.

When the text encoder outputs are cached (`--cache_text_encoder_outputs`), the training does not keep the text encoder
in memory, and the `sample` command can only use the prompts of the prompt file. This script loads the text encoder in
its own process, encodes the prompts, and sends the outputs to the training together with a `sample` command. The
settings of the text encoder are read from `manifest.json` of the run directory.

Unlike tools/run_control_cli.py, this script needs the Python environment of the training (PyTorch and so on). It
supports Anima (anima_train_network.py) and MOSS-Image (moss_image_train_network.py). See docs/run_control.md.

Usage:
    python tools/run_control_sample_prompt.py <run directory> --prompt "a cat --w 1024 --h 1024 --d 1"
    python tools/run_control_sample_prompt.py <run directory> --prompt "a cat" "a dog" --seed 42 --device cuda:1

Exit code: the same as tools/run_control_cli.py (0 = done, 1 = failed or not executed, 2 = timeout).
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TOOLS_DIR))  # the repository root, for `library`
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import run_control_cli  # noqa: E402

MANIFEST_FILE = "manifest.json"
TE_OUTPUTS_DIR = "te_outputs"  # in the commands directory


def collect_texts(prompts: List[str], negative_prompt: Optional[str]) -> List[str]:
    """The texts to encode: the prompts, their negative prompts, and the empty negative prompt."""
    from library import sampling

    texts: List[str] = []

    def add(text: Optional[str]):
        if text is not None and text not in texts:
            texts.append(text)

    for line in prompts:
        prompt_dict = sampling.line_to_prompt_dict(line)
        add(prompt_dict.get("prompt", ""))
        add(prompt_dict.get("negative_prompt"))
    add(negative_prompt)
    add("")  # the negative prompt when it is not given
    return texts


def encode_anima(train_args: dict, texts: List[str], device, blocks_to_swap: Optional[int]) -> Dict[str, list]:
    import torch

    from library import anima_utils, strategy_anima

    if blocks_to_swap:
        raise ValueError("--text_encoder_blocks_to_swap is not supported for Anima (the text encoder is small)")

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=train_args["qwen3"],
        t5_tokenizer_path=train_args.get("t5_tokenizer_path"),
        qwen3_max_length=train_args["qwen3_max_token_length"],
        t5_max_length=train_args["t5_max_token_length"],
    )
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(train_args.get("mixed_precision"), torch.float32)
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(train_args["qwen3"], dtype=dtype, device=device)
    text_encoder.eval()

    outputs = {}
    with torch.no_grad():
        for text in texts:
            tokens = tokenize_strategy.tokenize(text)
            outputs[text] = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    return outputs


def encode_moss_image(train_args: dict, texts: List[str], device, blocks_to_swap: Optional[int]) -> Dict[str, list]:
    import torch

    from library import moss_image_utils, strategy_moss_image

    tokenize_strategy = strategy_moss_image.MossImageTokenizeStrategy(
        train_args["moss_max_token_length"], train_args.get("tokenizer_cache_dir")
    )
    encoding_strategy = strategy_moss_image.MossImageTextEncodingStrategy()
    disable_mmap = bool(train_args.get("disable_mmap_load_safetensors"))

    if device.type == "cpu":
        if blocks_to_swap:
            raise ValueError("--text_encoder_blocks_to_swap needs a GPU (--device)")
        # float32 without ConvRot INT8, whatever the training uses
        text_encoder = moss_image_utils.load_text_encoder(
            train_args["text_encoder"], dtype=torch.float32, device="cpu", disable_mmap=disable_mmap
        )
    else:
        # the same as the training: bf16, ConvRot INT8 if the training uses it
        text_encoder = moss_image_utils.load_text_encoder(
            train_args["text_encoder"],
            dtype=torch.bfloat16,
            device="cpu",
            disable_mmap=disable_mmap,
            convrot_int8=bool(train_args.get("convrot_int8_text_encoder")),
            convrot_int8_bwd=train_args.get("convrot_int8_bwd") or "bf16",
            calc_device=device,
        )
        if blocks_to_swap:
            moss_image_utils.enable_text_encoder_block_swap(text_encoder, blocks_to_swap, device)
        else:
            text_encoder.to(device)
    text_encoder.eval()

    outputs = {}
    with torch.no_grad():
        for text in texts:
            tokens = tokenize_strategy.tokenize(text)
            outputs[text] = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    return outputs


# training script (`script` in manifest.json) -> function to encode the texts
ENCODERS: Dict[str, Callable] = {
    "anima_train_network.py": encode_anima,
    "moss_image_train_network.py": encode_moss_image,
}


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode new prompts with the text encoder in this process, and generate their sample images"
        " in a training running with --run_control"
    )
    parser.add_argument("run_dir", type=str, help="run directory of the training (<output_dir>/<output_name>_run)")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        required=True,
        help='prompts in the syntax of the prompt file, e.g. "a cat --w 1024 --h 1024 --d 1 --n low quality"',
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for all prompts of this command")
    parser.add_argument("--steps", type=int, default=None, help="number of sampling steps for all prompts of this command")
    parser.add_argument("--width", type=int, default=None, help="width for all prompts of this command")
    parser.add_argument("--height", type=int, default=None, help="height for all prompts of this command")
    parser.add_argument("--scale", type=float, default=None, help="CFG scale for all prompts of this command")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for all prompts of this command")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device for the text encoder (default cpu). The GPU of the training is in use: specify a GPU only if it has"
        " enough free memory. An error in this script does not affect the training",
    )
    parser.add_argument(
        "--text_encoder_blocks_to_swap",
        type=int,
        default=None,
        help="number of text encoder layers to stream from CPU, to reduce the VRAM (MOSS-Image with a GPU only)."
        " The default is the setting of the training",
    )
    parser.add_argument("--no_wait", action="store_true", help="do not wait for the result, only print the name of the command file")
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="seconds to wait for the result (default 3600). The command is not canceled"
    )
    return parser


def main() -> int:
    args = setup_parser().parse_args()
    args.command = "sample"
    args.index = None
    run_dir = os.path.abspath(args.run_dir)

    status = run_control_cli.read_json(os.path.join(run_dir, run_control_cli.STATUS_FILE))
    manifest = run_control_cli.read_json(os.path.join(run_dir, MANIFEST_FILE))
    if status is None or manifest is None:
        print(f"status.json or manifest.json is not found in {run_dir}. Is the training started with --run_control?", file=sys.stderr)
        return 1
    if status.get("state") in run_control_cli.FINAL_STATES:
        print(f"the training is not running (state: {status.get('state')})", file=sys.stderr)
        return 1

    command = run_control_cli.build_command(args)
    train_args = manifest.get("args") or {}

    if not train_args.get("cache_text_encoder_outputs"):
        # the text encoder is in the training process: the usual command is enough
        print("the text encoder outputs are not cached in this training. sending the usual sample command", file=sys.stderr)
    else:
        script = manifest.get("script")
        encoder = ENCODERS.get(script)
        if encoder is None:
            print(f"{script} is not supported. supported: {', '.join(ENCODERS.keys())}", file=sys.stderr)
            return 1
        if not train_args.get("sample_prompts"):
            print(
                "the training was started without the prompt file (--sample_prompts), so it does not accept text encoder outputs",
                file=sys.stderr,
            )
            return 1

        # relative paths in the arguments of the training are relative to its working directory
        cwd = manifest.get("cwd")
        if cwd and os.path.isdir(cwd):
            os.chdir(cwd)

        import torch

        from library import sampling

        blocks_to_swap = args.text_encoder_blocks_to_swap
        if blocks_to_swap is None and args.device != "cpu":
            blocks_to_swap = train_args.get("text_encoder_blocks_to_swap")

        texts = collect_texts(args.prompt, args.negative_prompt)
        start = time.time()
        try:
            outputs = encoder(train_args, texts, torch.device(args.device), blocks_to_swap)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"encoded {len(texts)} texts in {time.time() - start:.1f} seconds (including loading)", file=sys.stderr)

        te_outputs_dir = os.path.join(run_dir, run_control_cli.COMMANDS_DIR, TE_OUTPUTS_DIR)
        os.makedirs(te_outputs_dir, exist_ok=True)
        te_outputs_name = f"{time.time_ns()}.safetensors"
        sampling.save_prompt_te_outputs(os.path.join(te_outputs_dir, te_outputs_name), outputs)
        command["te_outputs"] = f"{TE_OUTPUTS_DIR}/{te_outputs_name}"  # relative to the commands directory

    name = run_control_cli.send_command(run_dir, command)
    if args.no_wait:
        print(json.dumps({"file": name}))
        return 0

    result = run_control_cli.wait_for_result(run_dir, name, args.timeout)
    if result is None:
        print(f"timeout: no result of {name} in {args.timeout} seconds. The command is not canceled", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
