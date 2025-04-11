import torch
import math
import warnings
from typing import Optional


def initialize_lora(lora_down: torch.nn.Module, lora_up: torch.nn.Module):
    torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(lora_up.weight)


# URAE: Ultra-Resolution Adaptation with Ease
def initialize_urae(
    org_module: torch.nn.Module,
    lora_down: torch.nn.Module,
    lora_up: torch.nn.Module,
    scale: float,
    rank: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    org_module_device = org_module.weight.device
    org_module_weight_dtype = org_module.weight.data.dtype
    org_module_requires_grad = org_module.weight.requires_grad

    device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    assert isinstance(device, torch.device), f"Invalid device type: {device}"

    weight = org_module.weight.data.to(device, dtype=torch.float32)

    with torch.autocast(device.type):
        # SVD decomposition
        V, S, Uh = torch.linalg.svd(weight, full_matrices=False)

        # For URAE, use the LAST/SMALLEST singular values and vectors (residual components)
        Vr = V[:, -rank:]
        Sr = S[-rank:]
        Sr /= rank
        Uhr = Uh[-rank:, :]

        # Create down and up matrices
        down = torch.diag(torch.sqrt(Sr)) @ Uhr
        up = Vr @ torch.diag(torch.sqrt(Sr))

        # Get expected shapes
        expected_down_shape = lora_down.weight.shape
        expected_up_shape = lora_up.weight.shape

        # Verify shapes match expected
        if down.shape != expected_down_shape:
            warnings.warn(UserWarning(f"Warning: Down matrix shape mismatch. Got {down.shape}, expected {expected_down_shape}"))

        if up.shape != expected_up_shape:
            warnings.warn(UserWarning(f"Warning: Up matrix shape mismatch. Got {up.shape}, expected {expected_up_shape}"))

        # Assign to LoRA weights
        lora_up.weight.data = up
        lora_down.weight.data = down

        # Optionally, subtract from original weight
        weight = weight - scale * (up @ down)
        org_module.weight.data = weight.to(org_module_device, dtype=org_module_weight_dtype)
        org_module.weight.requires_grad = org_module_requires_grad


# PiSSA: Principal Singular Values and Singular Vectors Adaptation
def initialize_pissa(
    org_module: torch.nn.Module,
    lora_down: torch.nn.Module,
    lora_up: torch.nn.Module,
    scale: float,
    rank: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    org_module_device = org_module.weight.device
    org_module_weight_dtype = org_module.weight.data.dtype
    org_module_requires_grad = org_module.weight.requires_grad

    device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    assert isinstance(device, torch.device), f"Invalid device type: {device}"

    weight = org_module.weight.data.clone().to(device, dtype=torch.float32)

    with torch.no_grad():
        # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
        V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
        Vr = V[:, :rank]
        Sr = S[:rank]
        Sr /= rank
        Uhr = Uh[:rank]

        down = torch.diag(torch.sqrt(Sr)) @ Uhr
        up = Vr @ torch.diag(torch.sqrt(Sr))

        # Get expected shapes
        expected_down_shape = lora_down.weight.shape
        expected_up_shape = lora_up.weight.shape

        # Verify shapes match expected or reshape appropriately
        if down.shape != expected_down_shape:
            warnings.warn(UserWarning(f"Down matrix shape mismatch. Got {down.shape}, expected {expected_down_shape}"))

        if up.shape != expected_up_shape:
            warnings.warn(UserWarning(f"Up matrix shape mismatch. Got {up.shape}, expected {expected_up_shape}"))

        lora_up.weight.data = up.to(dtype=lora_up.weight.dtype)
        lora_down.weight.data = down.to(dtype=lora_down.weight.dtype)

        weight = weight.data - scale * (up @ down)
        org_module.weight.data = weight.to(org_module_device, dtype=org_module_weight_dtype)
        org_module.weight.requires_grad = org_module_requires_grad

