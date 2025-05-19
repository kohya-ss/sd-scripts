import torch
import math
import warnings
from torch import Tensor
from typing import Optional
from library.incremental_pca import IncrementalPCA
from dataclasses import dataclass


@dataclass
class InitializeParams:
    """Parameters for initialization methods (PiSSA, URAE)"""

    use_ipca: bool = False
    use_lowrank: bool = False
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
    # Store original device, dtype, and requires_grad status
    orig_device = org_module.weight.device
    orig_dtype = org_module.weight.data.dtype
    orig_requires_grad = org_module.weight.requires_grad

    # Determine device and dtype to work with
    device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dtype = dtype if dtype is not None else lora_down.weight.data.dtype

    # Move original weight to chosen device and use float32 for numerical stability
    weight = org_module.weight.data.to(device, dtype=torch.float32)

    with torch.autocast(device.type), torch.no_grad():
        # Perform SVD decomposition (either directly or with IPCA for memory efficiency)
        if use_ipca:
            ipca = IncrementalPCA(
                n_components=None,
                batch_size=1024,
                lowrank=use_lowrank,
                lowrank_q=lowrank_q if lowrank_q is not None else min(weight.shape),
                lowrank_niter=lowrank_niter,
                lowrank_seed=lowrank_seed,
            )
            ipca.fit(weight)

            # Extract singular values and vectors, focusing on the minor components (smallest singular values)
            S_full = ipca.singular_values_
            V_full = ipca.components_.T  # Shape: [out_features, total_rank]

            # Get identity matrix to transform for right singular vectors
            identity = torch.eye(weight.shape[1], device=weight.device)
            Uhr_full = ipca.transform(identity).T  # Shape: [total_rank, in_features]

            # Extract the last 'rank' components (the minor/smallest ones)
            Sr = S_full[-rank:]
            Vr = V_full[:, -rank:]
            Uhr = Uhr_full[-rank:]

            # Scale singular values
            Sr = Sr / rank
        else:
            # Direct SVD approach
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

            # Extract the minor components (smallest singular values)
            Sr = S[-rank:]
            Vr = U[:, -rank:]
            Uhr = Vh[-rank:]

            # Scale singular values
            Sr = Sr / rank

        # Create the low-rank adapter matrices by splitting the minor components
        # Down matrix: scaled right singular vectors with singular values
        down_matrix = torch.diag(torch.sqrt(Sr)) @ Uhr

        # Up matrix: scaled left singular vectors with singular values
        up_matrix = Vr @ torch.diag(torch.sqrt(Sr))

    # Assign to LoRA modules
    lora_down.weight.data = down_matrix.to(device=device, dtype=dtype)
    lora_up.weight.data = up_matrix.to(device=device, dtype=dtype)

    # Update the original weight by removing the minor components
    # This is equivalent to keeping only the major components
    modified_weight = weight - scale * (up_matrix @ down_matrix)
    org_module.weight.data = modified_weight.to(device=orig_device, dtype=orig_dtype)
    org_module.weight.requires_grad = orig_requires_grad


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
    use_lowrank: bool = False,
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
            with torch.autocast(device.type, dtype=torch.float64):
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

        # Uhr may be in higher precision
        with torch.autocast(device.type, dtype=Uhr.dtype):
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


def convert_pissa_to_standard_lora(trained_up: Tensor, trained_down: Tensor, orig_up: Tensor, orig_down: Tensor, rank: int):
    with torch.no_grad():
        # Calculate ΔW = A'B' - AB
        delta_w = (trained_up @ trained_down) - (orig_up @ orig_down)

        # We need to create new low-rank matrices that represent this delta
        U, S, V = torch.linalg.svd(delta_w.to(device="cuda", dtype=torch.float32), full_matrices=False)

        # Take the top 2*r singular values (as suggested in the paper)
        rank = rank * 2
        rank = min(rank, len(S))  # Make sure we don't exceed available singular values

        # Create new LoRA matrices
        new_up = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        new_down = torch.diag(torch.sqrt(S[:rank])) @ V[:rank, :]

    # These matrices can now be used as standard LoRA weights
    return new_up, new_down


def convert_urae_to_standard_lora(
    trained_up: Tensor,
    trained_down: Tensor,
    orig_up: Tensor,
    orig_down: Tensor,
    initial_alpha: float | None = None,
    rank: int | None = None,
):
    """
    Convert URAE trained weights to standard LoRA format

    Args:
        trained_up: The trained URAE Up matrix
        trained_down: The trained URAE Down matrix
        orig_up: The original up matrix before training
        orig_down: The original down matrix before training
        initial_alpha: The alpha value used during URAE training (if any)
        rank: The rank for the standard LoRA (if None, uses the rank of trained_A)

    Returns:
        lora_up: Standard LoRA up matrix
        lora_down: Standard LoRA down matrix
        alpha: Appropriate alpha value for the LoRA
    """
    with torch.no_grad():
        # Calculate the weight delta
        delta_w = (trained_up @ trained_down) - (orig_up @ orig_down)

        # Perform SVD on the delta
        U, S, V = torch.linalg.svd(delta_w.to(dtype=torch.float32), full_matrices=False)

        # If rank is not specified, use the same rank as the trained matrices
        if rank is None:
            rank = trained_up.shape[1]
        else:
            # Ensure we don't exceed available singular values
            rank = min(rank, len(S))

        # Create standard LoRA matrices using top singular values
        # This is now standard LoRA (using top values), not URAE (which used bottom values during training)
        lora_up = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        lora_down = torch.diag(torch.sqrt(S[:rank])) @ V[:rank, :]

        # Method 1: Preserve the Frobenius norm of the delta
        original_effect: float = torch.norm(delta_w, p="fro").item()
        unscaled_lora_effect: float = torch.norm(lora_up @ lora_down, p="fro").item()

        # The scaling factor in lora is (alpha/r), so:
        # alpha/r × ||AB|| = ||delta_W||
        # alpha = r × ||delta_W|| / ||AB||
        if unscaled_lora_effect > 0:
            norm_based_alpha = rank * (original_effect / unscaled_lora_effect)
        else:
            norm_based_alpha = 1.0  # Fallback

        # Method 2: If initial_alpha is provided, adjust based on rank change
        if initial_alpha is not None:
            initial_rank = trained_up.shape[1]
            # Scale alpha proportionally if rank changed
            rank_adjusted_alpha = initial_alpha * (rank / initial_rank)
        else:
            rank_adjusted_alpha = None

        # Choose the appropriate alpha
        if rank_adjusted_alpha is not None:
            # Use the rank-adjusted alpha, but ensure it's not too different from norm-based
            # Cap the difference to avoid extreme values
            alpha = rank_adjusted_alpha
            # Optional: Cap alpha to be within a reasonable range of norm_based_alpha
            if norm_based_alpha > 0:
                max_factor = 5.0  # Allow up to 5x difference
                upper_bound = norm_based_alpha * max_factor
                lower_bound = norm_based_alpha / max_factor
                alpha = min(max(alpha, lower_bound), upper_bound)
        else:
            # Use norm-based alpha
            alpha = norm_based_alpha

        # Round to a clean value for better usability
        alpha = round(alpha, 2)

        # Ensure alpha is positive and within reasonable bounds
        alpha = max(0.1, min(alpha, 1024.0))

    return lora_up, lora_down, alpha
