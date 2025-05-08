import torch
import math
import warnings
from typing import Optional
from library.incremental_pca import IncrementalPCA
from dataclasses import dataclass


@dataclass
class InitializeParams:
    """Parameters for initialization methods (PiSSA, URAE)"""

    use_ipca: bool = False
    use_lowrank: bool = True
    lowrank_q: Optional[int] = None
    lowrank_niter: int = 4
    lowrank_seed: Optional[int] = None


def initialize_parse_opts(key: str) -> InitializeParams:
    """
    Parse initialization parameters from a string key.

    Format examples:
    - "pissa" -> Default PiSSA with lowrank=True, niter=4
    - "pissa_niter_4" -> PiSSA with niter=4
    - "pissa_lowrank_false" -> PiSSA without lowrank
    - "pissa_ipca_true" -> PiSSA with IPCA
    - "pissa_q_16" -> PiSSA with lowrank_q=16
    - "pissa_seed_42" -> PiSSA with seed=42
    - "urae_..." -> Same options but for URAE

    Args:
        key: String key to parse

    Returns:
        InitializeParams object with parsed parameters
    """
    parts = key.lower().split("_")

    # Extract the method (first part)
    method = parts[0]
    if method not in ["pissa", "urae"]:
        raise ValueError(f"Unknown initialization method: {method}")

    # Start with default parameters
    params = InitializeParams()

    # Parse the remaining parts
    i = 1
    while i < len(parts):
        if parts[i] == "ipca":
            if i + 1 < len(parts) and parts[i + 1] in ["true", "false"]:
                params.use_ipca = parts[i + 1] == "true"
                i += 2
            else:
                params.use_ipca = True
                i += 1
        elif parts[i] == "lowrank":
            if i + 1 < len(parts) and parts[i + 1] in ["true", "false"]:
                params.use_lowrank = parts[i + 1] == "true"
                i += 2
            else:
                params.use_lowrank = True
                i += 1
        elif parts[i] == "niter":
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                params.lowrank_niter = int(parts[i + 1])
                i += 2
            else:
                i += 1
        elif parts[i] == "q":
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                params.lowrank_q = int(parts[i + 1])
                i += 2
            else:
                i += 1
        elif parts[i] == "seed":
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                params.lowrank_seed = int(parts[i + 1])
                i += 2
            else:
                i += 1
        else:
            # Skip unknown parameter
            i += 1

    return params


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
    use_ipca: bool = False,
    use_lowrank: bool = True,
    lowrank_q: Optional[int] = None,
    lowrank_niter: int = 4,
    lowrank_seed: Optional[int] = None,
):
    org_module_device = org_module.weight.device
    org_module_weight_dtype = org_module.weight.data.dtype
    org_module_requires_grad = org_module.weight.requires_grad

    dtype = dtype if dtype is not None else lora_down.weight.data.dtype
    device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    assert isinstance(device, torch.device), f"Invalid device type: {device}"

    weight = org_module.weight.data.to(device, dtype=torch.float32)

    if use_ipca:
        # For URAE we need all components to get the "residual" ones
        ipca = IncrementalPCA(
            n_components=None,  # Get all components
            batch_size=1024,
            lowrank=use_lowrank,
            lowrank_q=lowrank_q if lowrank_q is not None else min(weight.shape),  # Use full rank for accurate residuals
            lowrank_niter=lowrank_niter,
            lowrank_seed=lowrank_seed,
        )
        ipca.fit(weight)

        # For URAE, use the LAST/SMALLEST singular values
        total_rank = min(weight.shape[0], weight.shape[1])
        V_full = ipca.components_.T  # [out_features, total_rank]
        S_full = ipca.singular_values_  # [total_rank]

        # Get the smallest singular values and vectors
        Vr = V_full[:, -rank:]  # Last rank left singular vectors
        Sr = S_full[-rank:]  # Last rank singular values
        Sr /= rank

        # To get Uhr (last rank right singular vectors), transform basis vectors
        identity = torch.eye(weight.shape[1], device=weight.device)
        Uhr_full = ipca.transform(identity).T  # [total_rank, in_features]
        Uhr = Uhr_full[-rank:]  # Last rank right singular vectors
    else:
        # Standard SVD approach
        V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
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
    use_ipca: bool = False,
    use_lowrank: bool = True,
    lowrank_q: Optional[int] = None,
    lowrank_niter: int = 4,
    lowrank_seed: Optional[int] = None,
):
    org_module_device = org_module.weight.device
    org_module_weight_dtype = org_module.weight.data.dtype
    org_module_requires_grad = org_module.weight.requires_grad

    dtype = dtype if dtype is not None else lora_down.weight.data.dtype
    device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    assert isinstance(device, torch.device), f"Invalid device type: {device}"

    weight = org_module.weight.data.clone().to(device, dtype=torch.float32)

    with torch.no_grad():
        if use_ipca:
            # Use Incremental PCA for large matrices
            ipca = IncrementalPCA(
                n_components=rank,
                batch_size=1024,
                lowrank=use_lowrank,
                lowrank_q=lowrank_q if lowrank_q is not None else 2 * rank,
                lowrank_niter=lowrank_niter,
                lowrank_seed=lowrank_seed,
            )
            ipca.fit(weight)

            # Extract principal components and singular values
            Vr = ipca.components_.T  # [out_features, rank]
            Sr = ipca.singular_values_  # [rank]
            Sr /= rank

            # We need to get Uhr from transforming an identity matrix
            identity = torch.eye(weight.shape[1], device=weight.device)
            Uhr = ipca.transform(identity).T  # [rank, in_features]

        elif use_lowrank:
            # Use low-rank SVD approximation which is faster
            seed_enabled = lowrank_seed is not None
            q_value = lowrank_q if lowrank_q is not None else 2 * rank

            with torch.random.fork_rng(enabled=seed_enabled):
                if seed_enabled:
                    torch.manual_seed(lowrank_seed)
                U, S, V = torch.svd_lowrank(weight, q=q_value, niter=lowrank_niter)

            Vr = U[:, :rank]  # First rank left singular vectors
            Sr = S[:rank]  # First rank singular values
            Sr /= rank
            Uhr = V[:rank]  # First rank right singular vectors

        else:
            # Standard SVD approach
            V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
            Vr = V[:, :rank]
            Sr = S[:rank]
            Sr /= rank
            Uhr = Uh[:rank]

        # Create down and up matrices
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

        lora_up.weight.data = up.to(lora_up.weight.data.device, dtype=lora_up.weight.dtype)
        lora_down.weight.data = down.to(lora_down.weight.data.device, dtype=lora_down.weight.dtype)

        weight = weight.data - scale * (up @ down)
        org_module.weight.data = weight.to(org_module_device, dtype=org_module_weight_dtype)
        org_module.weight.requires_grad = org_module_requires_grad

