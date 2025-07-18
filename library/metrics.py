import torch
from collections import defaultdict
import numpy as np

# metrics.py

import torch
from collections import defaultdict
import numpy as np

def calculate_nfn_scores(model, batch, random_baseline=True):
    """
    Calculate NFN scores for all weight matrices.
    Args:
        model: Model to calculate NFN scores for.
        batch: Batch of problems.
        random_baseline: Whether to calculate the random baseline (this is True by default since it's needed for the NFN score).
    Returns:
        Dictionary of NFN scores for all weight matrices.
    """
    # Move batch to GPU if needed
    """ if next(model.parameters()).device != batch['sample'].device:
        batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} """
    if 'x' in batch and next(model.parameters()).device != batch['x'].device:
        batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'added_cond_kwargs' in batch:
            batch['added_cond_kwargs'] = {k: v.to(next(model.parameters()).device) for k, v in batch['added_cond_kwargs'].items()}

    # Initialize metrics dictionary
    metrics = defaultdict(dict)
    
    # Define hook function to calculate NFN scores for each weight matrix
    def hook_fn(name):
        def hook(module, input, output):
            if hasattr(module, 'weight') and module.weight is not None:
                z = input[0] if isinstance(input, tuple) else input
                W = module.weight
                
                # Check for FP8 weights and upcast if necessary
                if W.dtype == torch.float8_e4m3fn:
                    W = W.to(torch.float16)
                    z = z.to(torch.float16)
                else:
                    z = z.float()
                    W = W.float()
                
                if len(z.shape) > 2:
                    # For Conv2d, activations are (N, C, H, W). We flatten H and W.
                    # For Linear in Transformers, they are (N, SeqLen, Dim). We flatten N and SeqLen.
                    if isinstance(module, torch.nn.Conv2d):
                        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
                    else: # Linear
                        z = z.reshape(-1, z.shape[-1])
                
                # Handle cases where input or weight is empty
                if z.numel() == 0 or W.numel() == 0:
                    return

                try:
                    # We calculate the Frobenius norm of W to normalize W for stability, but it is not necessary.
                    W_norm = torch.norm(W.reshape(W.shape[0], -1), 'fro')
                    z_norm = torch.norm(z, dim=1, keepdim=True)

                    W_normalized = W / (W_norm + 1e-8)
                    z_normalized = z / (z_norm + 1e-8)
                    
                    if isinstance(module, torch.nn.Conv2d):
                         # Reshape W for matmul with flattened z
                        W_reshaped = W_normalized.reshape(W_normalized.shape[0], -1).t()
                        Wz = torch.mm(z_normalized, W_reshaped)
                    else: # Linear
                        Wz = torch.mm(z_normalized, W_normalized.t())
                        
                    metrics[name]['actual'] = torch.norm(Wz, dim=1).mean().item()/np.sqrt(z.shape[1])

                    if random_baseline:
                        z_random = torch.randn_like(z_normalized)
                        z_random_norm = torch.norm(z_random, dim=1, keepdim=True)
                        z_random_normalized = z_random / (z_random_norm + 1e-8)
                        
                        if isinstance(module, torch.nn.Conv2d):
                            Wz_random = torch.mm(z_random_normalized, W_reshaped)
                        else:
                            Wz_random = torch.mm(z_random_normalized, W_normalized.t())
                        
                        metrics[name]['random'] = torch.norm(Wz_random, dim=1).mean().item()/np.sqrt(z.shape[1])
                    
                    metrics[name]['nfn'] = metrics[name]['actual']/metrics[name]['random']
                except RuntimeError as e:
                    print(f"Error in layer {name}: {e}")
                    print(f"Input shape: {z.shape}, Weight shape: {W.shape}")
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        # Target Linear and Conv2d layers, excluding normalization and embedding layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if 'norm' not in name.lower() and 'embedding' not in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        _ = model(**batch)
    """ with torch.no_grad():
        # The SdxlUNet2DConditionModel expects positional arguments, not a dict unpack.
        # We must match the forward signature: forward(self, x, timesteps=None, context=None, y=None, **kwargs)
        x = batch.get('x')
        timesteps = batch.get('timesteps')
        context = batch.get('context')
        y = batch.get('y')

        # This handles both SDXL and non-SDXL cases based on what keys are present
        if y is not None: # SDXL
            _ = model(x, timesteps, context, y)
        else: # SD 1.5
            _ = model(x, timesteps, context) """
        
    for hook in hooks:
        hook.remove()

    return metrics


def average_metrics(metrics_list):
    """
    Averages a list of metric dictionaries.
    Each dictionary in the list is expected to have a structure like:
    { 'module_name': {'actual': float, 'random': float, 'nfn': float}, ... }
    """
    if not metrics_list:
        return {}
    
    sum_metrics = defaultdict(lambda: {'actual': 0.0, 'random': 0.0, 'nfn': 0.0, 'count': 0})

    for metrics_dict in metrics_list:
        for name, values in metrics_dict.items():
            sum_metrics[name]['actual'] += values.get('actual', 0.0)
            sum_metrics[name]['random'] += values.get('random', 0.0)
            sum_metrics[name]['nfn'] += values.get('nfn', 0.0)
            sum_metrics[name]['count'] += 1

    avg_metrics = {}
    for name, data in sum_metrics.items():
        count = data['count']
        if count > 0:
            avg_metrics[name] = {
                'actual': data['actual'] / count,
                'random': data['random'] / count,
                'nfn': data['nfn'] / count
            }
            
    return avg_metrics


def print_stylish_results(results, title="Results"):
    """Prints a formatted table of results."""
    max_key_len = max(len(k) for k in results.keys()) if results else 20
    header_width = max_key_len + 30
    
    print("\n" + "=" * header_width)
    print(f"{title:^{header_width}}")
    print("=" * header_width)
    print(f"{'Module/Block':<{max_key_len}} | {'NFN Score':>10} | {'Module Count':>12}")
    print("-" * header_width)
    
    for k, v in results.items():
        nfn_str = f"{v.get('nfn', 0.0):.4f}"
        count_str = str(v.get('module_count', 'N/A'))
        print(f"{k:<{max_key_len}} | {nfn_str:>10} | {count_str:>12}")
        
    print("=" * header_width + "\n")

def get_group_metrics(metrics, groups=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'down_proj'], individual=False):
    """
    Calculate group metrics.
    Args:
        metrics: Dictionary of NFN scores for all weight matrices.
        groups: List of groups to calculate metrics for.
    Returns:
        Dictionary of group metrics.
    """
    group_metrics = defaultdict(dict)
    for group in groups:
        group_metrics[group] = {
            'count': 0,
            'actual_sum': 0.0,
            'random_sum': 0.0,
            'nfn_sum': 0.0
        }
    for name, values in metrics.items():
        for group in groups:
            if group in name:
                group_metrics[group]['count'] += 1
                group_metrics[group]['actual_sum'] += values.get('actual', 0.0)
                group_metrics[group]['random_sum'] += values.get('random', 0.0)
                group_metrics[group]['nfn_sum'] += values.get('nfn', 0.0)
    results = {}
    for group, data in group_metrics.items():
        count = data['count']
        if count > 0:
            if not individual:
                results[group] = {
                'actual': data['actual_sum'] / count,
                    'random': data['random_sum'] / count if 'random_sum' in data else 0.0,
                    'nfn': data['actual_sum'] / data['random_sum'] if 'random_sum' in data else 0.0
                }
            else:
                results[group] = {
                    'actual': data['actual_sum'] / count,
                    'random': data['random_sum'] / count if 'random_sum' in data else 0.0,
                    'nfn': data['nfn_sum'] / count
                }
        else:
            results[group] = {'actual': 0.0, 'random': 0.0, 'nfn': 0.0}
    return results 