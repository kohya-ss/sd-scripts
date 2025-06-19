from contextlib import contextmanager
import torch
import logging

logger = logging.getLogger(__name__)


def maybe_sample_params(optimizer):
    """
    Returns parameter sampling context for IVON optimizers, otherwise returns no-op context.

    pip install ivon-opt

    Args:
        optimizer: PyTorch optimizer instance.

    Returns:
        Context manager for parameter sampling if optimizer supports it, otherwise nullcontext().
    """
    from contextlib import nullcontext

    return optimizer.sampled_params(train=True) if hasattr(optimizer, "sampled_params") else nullcontext()


@contextmanager
def maybe_pruned_save(model, optimizer, enable_pruning=False, pruning_ratio=0.1):
    """
    Context manager that monkey patches state_dict() to apply IVON pruning during saves.

    Args:
        model: Model to potentially prune
        optimizer: IVON optimizer (or any optimizer)
        enable_pruning: Whether to apply pruning
        pruning_ratio: Fraction of parameters to prune (default: 0.1)

    Usage:
        with maybe_pruned_save(model, optimizer, enable_pruning=True):
            model.save_weights(...)  # Saved state_dict will have pruned weights
        # Model's state_dict is automatically restored after save
    """
    # Check if we should prune - more flexible detection of IVON-like optimizers
    should_prune = enable_pruning and (
        hasattr(optimizer, "sampled_params")
    )

    if not should_prune:
        yield
        return

    param_variances = []

    # Extract variances from IVON optimizer
    offset = 0
    for group in optimizer.param_groups:
        # Get group-level values
        ess = group["ess"]          # λ (lambda)
        weight_decay = group["weight_decay"]  # δ (delta)
        hess = group["hess"]        # hᵢ (Hessian diagonal)
        
        # Calculate variance: vᵢ = 1 / (λ × (hᵢ + δ))
        group_variance = 1.0 / (ess * (hess + weight_decay))
        
        # Map back to individual parameters
        param_offset = 0
        for param in group["params"]:
            if param is not None and param.requires_grad:
                param_numel = param.numel()
                param_slice = slice(param_offset, param_offset + param_numel)
                
                # Get variance for this parameter
                param_var = group_variance[param_slice]
                
                # Store each element's variance with its location
                flat_param_var = param_var.view(-1)
                for i, var_val in enumerate(flat_param_var):
                    param_variances.append((var_val.item(), param, i))
                
                param_offset += param_numel
        
        offset += group["numel"]
    
    if not param_variances:
        yield
        return
    
    param_variances.sort(key=lambda x: x[0], reverse=True)  # Highest variance first
    num_to_prune = int(len(param_variances) * pruning_ratio)

    pruning_mask = {}

    # Build mask for each parameter
    for param in model.parameters():
        pruning_mask[id(param)] = torch.ones_like(param, dtype=torch.bool)

    # Mark parameters to prune
    for param in model.parameters():
        mask = pruning_mask[id(param)]
        num_to_prune = int(mask.numel() * pruning_ratio)

        # Flatten and create indices to zero out
        flat_mask = mask.view(-1)
        prune_indices = torch.randperm(flat_mask.numel())[:num_to_prune]
        flat_mask[prune_indices] = False

        # Restore original mask shape
        pruning_mask[id(param)] = flat_mask.view(mask.shape)

    # Monkey patch state_dict
    original_state_dict = model.state_dict

    def pruned_state_dict(*args, **kwargs):
        state_dict = original_state_dict(*args, **kwargs)
        for name, param in model.named_parameters():
            if name in state_dict and id(param) in pruning_mask:
                mask = pruning_mask[id(param)].to(state_dict[name].device)
                state_dict[name] = state_dict[name] * mask.float()
        return state_dict

    model.state_dict = pruned_state_dict

    try:
        pruned_count = sum(1 for mask in pruning_mask.values() for val in mask.flatten() if not val)
        total_params = sum(mask.numel() for mask in pruning_mask.values())
        logger.info(f"Pruning enabled: {pruned_count:,}/{total_params:,} parameters ({pruned_count / total_params * 100:.1f}%)")
        yield
    finally:
        # Restore original state_dict
        model.state_dict = original_state_dict
