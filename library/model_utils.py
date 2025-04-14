import torch
from torch import nn, Tensor
import torch.nn.functional as F


class AID(nn.Module):
    def __init__(self, dropout_prob=0.9):
        super().__init__()
        self.p = dropout_prob

    def forward(self, x: Tensor):
        if self.training:
            # Use boolean masks and torch.where for better efficiency
            pos_mask = x > 0

            # Process positive values (keep with probability p)
            pos_vals = torch.where(pos_mask, x, torch.zeros_like(x))
            pos_dropped = F.dropout(pos_vals, p=1 - self.p, training=True)
            if self.p > 0:
                pos_dropped = pos_dropped / self.p

            # Process negative values (keep with probability 1-p)
            neg_vals = torch.where(~pos_mask, x, torch.zeros_like(x))
            neg_dropped = F.dropout(neg_vals, p=self.p, training=True)
            if (1 - self.p) > 0:
                neg_dropped = neg_dropped / (1 - self.p)

            return pos_dropped + neg_dropped
        else:
            # Simplified test-time behavior
            return torch.where(x > 0, self.p * x, (1 - self.p) * (-x))


class AID_GELU(nn.Module):
    def __init__(self, dropout_prob=0.9, approximate="none"):
        super().__init__()
        self.p = dropout_prob
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        # Apply GELU first
        x = self.gelu(x)

        if self.training:
            # Create masks once and reuse
            pos_mask = x > 0

            # Process positive values (keep with probability p)
            pos_vals = torch.where(pos_mask, x, torch.zeros_like(x))
            pos_dropped = F.dropout(pos_vals, p=1 - self.p, training=True)
            if self.p > 0:
                pos_dropped = pos_dropped / self.p

            # Process negative values (keep with probability 1-p)
            neg_vals = torch.where(~pos_mask, x, torch.zeros_like(x))
            neg_dropped = F.dropout(neg_vals, p=self.p, training=True)
            if (1 - self.p) > 0:
                neg_dropped = neg_dropped / (1 - self.p)

            return pos_dropped + neg_dropped
        else:
            # Test time behavior - simplify with direct where operations
            return torch.where(x > 0, self.p * x, (1 - self.p) * x)
