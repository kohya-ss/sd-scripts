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
