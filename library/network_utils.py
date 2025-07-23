import torch
from torch import Tensor

# LoRA Dropout as a Sparsity Regularizer for Overfitting Control
def lora_dropout_down(down: Tensor, x: Tensor, dropout_prob=0.5):
    """ A = A · diag(mA), mA ∼ Bern(1 − p)"""
    mask = torch.bernoulli(
        torch.ones(down.shape[1], device=down.device) * (1 - dropout_prob)
    )
    
    # Apply input dimension mask (columns of down-projection)
    lx = x @ (down * mask.view(1, -1)).t()
    return lx

def lora_dropout_up(up: Tensor, x: Tensor, dropout_prob=0.5):
    """ B = B⊤ · diag(mB )⊤ , mB ∼ Bern(1 − p)"""
    mask = torch.bernoulli(
        torch.ones(up.shape[0], device=up.device) * (1 - dropout_prob)
    )

    # Apply output dimension mask (rows of up-projection)
    lx = x @ (up * mask.view(-1, 1)).t()
    return lx
