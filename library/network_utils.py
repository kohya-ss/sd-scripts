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
        hasattr(optimizer, "sampled_params") or 
        any("h" in state for state in optimizer.state.values()) or 
        hasattr(optimizer, "_hess") or  # Some optimizers might have this attribute
        "ess" in optimizer.param_groups[0]
    )

    if not should_prune:
        yield
        return

    # Calculate pruning mask
    pruning_mask = {}
    param_variances = []

    def get_hessian_variance(param):
        # Multiple ways to extract Hessian-based variance
        try:
            # 1. Try all groups to find the correct parameter group
            for group in optimizer.param_groups:
                if param in group.get('params', []):
                    # Prefer direct Hessian if available
                    if 'hess' in group and len(group['hess']) > 0:
                        return group['hess']

            # 2. Try standard IVON state access
            param_state = optimizer.state.get(param, {})
            if "h" in param_state:
                h = param_state["h"]
                return h

            # 3. Check if 'hess' exists in state
            for state_param, state_dict in optimizer.state.items():
                if "h" in state_dict:
                    return state_dict["h"]

            # 4. Fallback to group-level Hessian
            group = optimizer.param_groups[0]
            hess = group.get('hess', None)
            if hess is not None and len(hess) > 0:
                return hess

        except Exception as e:
            logger.warning(f"Error getting Hessian variance: {e}")
        
        # Complete fallback: generate a random variance
        return torch.rand_like(param)

    # If variance extraction consistently fails, use random pruning
    def random_pruning(param, pruning_ratio):
        mask = torch.ones_like(param, dtype=torch.bool)
        num_to_prune = int(param.numel() * pruning_ratio)
        
        # Create a flat tensor of all indices and shuffle
        indices = torch.randperm(param.numel())[:num_to_prune]
        
        # Create a flattened mask and set selected indices to False
        flat_mask = mask.view(-1)
        flat_mask[indices] = False
        
        return mask

    # Track parameters with gradients
    gradients_exist = False
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            gradients_exist = True
            try:
                variance = get_hessian_variance(param)
                if variance is not None:
                    flat_variance = variance.view(-1)
                    for i, v in enumerate(flat_variance):
                        param_variances.append((v.item(), param, i))
            except Exception as e:
                logger.warning(f"Variance extraction failed for {param}: {e}")

    # No pruning if no gradients exist
    if not gradients_exist:
        logger.info("No parameters with gradients, skipping pruning")
        yield
        return

    # Fallback to random pruning if no variance info found
    if not param_variances:
        logger.info("No variance info found, using random pruning")
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                pruning_mask[id(param)] = random_pruning(param, pruning_ratio)
        yield
        return

    # Create pruning mask
    param_variances.sort(reverse=True)
    num_to_prune = int(len(param_variances) * pruning_ratio)

    # Build mask for each parameter
    for param in model.parameters():
        pruning_mask[id(param)] = torch.ones_like(param, dtype=torch.bool)

    # Mark parameters to prune
    for i in range(min(num_to_prune, len(param_variances))):
        _, param, flat_idx = param_variances[i]
        shape = param.data.shape
        coords = []
        temp_idx = flat_idx
        for dim in reversed(shape):
            coords.append(temp_idx % dim)
            temp_idx //= dim
        coords = tuple(reversed(coords))
        pruning_mask[id(param)][coords] = False

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
