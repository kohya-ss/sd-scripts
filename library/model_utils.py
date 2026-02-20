import torch
from torch import nn, Tensor


class AID(nn.Module):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor):
        if self.training:
            pos_mask = (x >= 0) * torch.bernoulli(torch.ones_like(x) * self.p)
            neg_mask = (x < 0) * torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            return x * (pos_mask + neg_mask)
        else:
            pos_part = (x >= 0) * x * self.p
            neg_part = (x < 0) * x * (1 - self.p)
            return pos_part + neg_part
