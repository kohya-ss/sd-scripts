import torch 

def generate_synthetic_weights(org_weight, seed=42):
    generator = torch.manual_seed(seed)
    
    # Base random normal distribution
    weights = torch.randn_like(org_weight)
    
    # Add structured variance to mimic real-world weight matrices
    # Techniques to create more realistic weight distributions:
    
    # 1. Block-wise variation
    block_size = max(1, org_weight.shape[0] // 4)
    for i in range(0, org_weight.shape[0], block_size):
        block_end = min(i + block_size, org_weight.shape[0])
        block_variation = torch.randn(1, generator=generator) * 0.3  # Local scaling
        weights[i:block_end, :] *= (1 + block_variation)
    
    # 2. Sparse connectivity simulation
    sparsity_mask = torch.rand(org_weight.shape, generator=generator) > 0.2  # 20% sparsity
    weights *= sparsity_mask.float()
    
    # 3. Magnitude decay
    magnitude_decay = torch.linspace(1.0, 0.5, org_weight.shape[0]).unsqueeze(1)
    weights *= magnitude_decay
    
    # 4. Add structured noise
    structural_noise = torch.randn_like(org_weight) * 0.1
    weights += structural_noise
    
    # Normalize to have similar statistical properties to trained weights
    weights = (weights - weights.mean()) / weights.std()
    
    return weights
