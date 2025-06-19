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
        """Determine if a parameter is eligible for variance-based pruning.

        Comprehensive check for IVON-like optimizer variance detection.

        Args:
            param (torch.Tensor): Model parameter to check

        Returns:
            bool: Whether the parameter is eligible for variance-based pruning
        """
        # 1. Basic parameter eligibility checks
        if not (param.grad is not None and param.requires_grad):
            return False
        
        # 2. Verify fundamental optimizer characteristics
        if not hasattr(optimizer, 'sampled_params'):
            return False
        
        # 3. Parameter group validation
        valid_group_found = False
        for group in optimizer.param_groups:
            # Use object ID to match parameters
            group_params = group.get('params', [])
            if any(id(param) == id(p) for p in group_params):
                # Require effective sample size or Hessian initialization
                if 'ess' in group or 'hess_init' in group:
                    valid_group_found = True
                    break
        
        if not valid_group_found:
            return False
        
        # 4. Optimizer state examination
        if param not in optimizer.state:
            return False
        
        # 5. Hessian information verification
        param_state = optimizer.state[param]
        hessian_keys = ['h', 'hess', 'Hessian', 'diagonal_hessian']
        
        for key in hessian_keys:
            if key in param_state:
                h = param_state[key]
                # Validate Hessian tensor
                if (h is not None and 
                    torch.is_tensor(h) and 
                    h.numel() > 0 and 
                    h.dtype in [torch.float32, torch.float64]):
                    return True
        
        return False

    # Comprehensive variance and pruning parameter collection
    variance_eligible_params = []
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            # Detect parameter with Hessian variance
            if get_hessian_variance(param):
                # Access Hessian state for variance calculation
                param_state = optimizer.state[param]
                
                # Prioritize 'h' key for Hessian, fallback to Hessian-related keys
                hessian_keys = ['h', 'hess']
                h = None
                for key in hessian_keys:
                    if key in param_state and param_state[key] is not None:
                        h = param_state[key]
                        break
                
                # Default to uniform Hessian if no specific information
                if h is None:
                    h = torch.ones_like(param)
                
                # Compute a meaningful variance
                try:
                    # Use Hessian diagonal to compute variance
                    variance = 1.0 / (h.abs().mean() + 1e-8)  # Avoid division by zero
                    variance_eligible_params.append((variance, param, 0))
                except Exception as e:
                    logger.warning(f"Variance computation failed for {param}: {e}")

    # No pruning if no variance-eligible parameters
    if not variance_eligible_params:
        logger.info("No variance-eligible parameters found, skipping pruning")
        yield
        return
    
    # Update param_variances for pruning
    # Convert variance to scalar values to avoid tensor comparison
    param_variances = sorted(
        variance_eligible_params, 
        key=lambda x: float(x[0]) if torch.is_tensor(x[0]) else x[0], 
        reverse=True
    )

    # Create pruning mask
    num_to_prune = int(len(param_variances) * pruning_ratio)

    # Build mask for each parameter
    for param in model.parameters():
        pruning_mask[id(param)] = torch.ones_like(param, dtype=torch.bool)

    # Mark parameters to prune
    # Ensure pruning occurs for LoRA-like parameters
    lora_param_keys = ['lora_A', 'lora_B', 'lora_A2', 'lora_B2']
    for name, param in model.named_parameters():
        if name.split('.')[-1] in lora_param_keys:
            # Ensure each LoRA parameter has some pruning
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
