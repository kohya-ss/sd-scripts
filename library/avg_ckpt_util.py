from typing import Dict, List, Optional
import os
import re
import torch
from safetensors.torch import load_file as load_safetensors


def filter_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only LoRA weights from state_dict on CPU as float32."""
    return {k: v.detach().float().cpu() for k, v in state_dict.items() if k.startswith("lora_")}


def average_state_dicts(states: List[Dict[str, torch.Tensor]], mode: str = "uniform", metrics: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    assert len(states) > 0
    keys = states[0].keys()
    if mode == "metric" and metrics is None:
        metrics = [1.0] * len(states)
    if mode == "uniform":
        avg = {k: torch.stack([sd[k] for sd in states], dim=0).mean(dim=0) for k in keys}
    elif mode == "ema":
        alpha = 2 / (len(states) + 1)
        avg = {k: states[0][k].clone() for k in keys}
        for sd in states[1:]:
            for k in keys:
                avg[k].mul_(1 - alpha).add_(sd[k], alpha=alpha)
    else:  # metric
        total = sum(metrics)
        weights = [m / total for m in metrics]
        avg = {k: sum(w * sd[k] for sd, w in zip(states, weights)) for k in keys}
    return avg


def load_lora_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, map_location="cpu")
    return filter_lora_state_dict(sd)


def collect_last_checkpoints(output_dir: str, model_name: str, ext: str, n: int) -> List[str]:
    pattern = re.compile(re.escape(model_name) + r"-(\d{6})" + re.escape(ext) + "$")
    files = []
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), os.path.join(output_dir, f)))
    files.sort()
    return [p for _, p in files[-n:]]
