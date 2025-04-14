from torch import nn, Tensor
import torch.nn.functional as F


class AID(nn.Module):
    def __init__(self, dropout_prob=0.9):
        super().__init__()
        self.p = dropout_prob

    def forward(self, x: Tensor):
        if self.training:
            # Create separate tensors for positive and negative components
            pos_mask = (x > 0).float()
            neg_mask = (x <= 0).float()

            pos_vals = x * pos_mask
            neg_vals = x * neg_mask

            # Apply dropout directly with PyTorch's F.dropout
            pos_dropped = F.dropout(pos_vals, p=1 - self.p, training=True)
            if self.p > 0:
                pos_dropped = pos_dropped / self.p  # Scale to maintain expectation

            neg_dropped = F.dropout(neg_vals, p=self.p, training=True)
            if (1 - self.p) > 0:
                neg_dropped = neg_dropped / (1 - self.p)  # Scale to maintain expectation

            # Combine results
            return pos_dropped + neg_dropped
        else:
            # During testing, use modified leaky ReLU with coefficient p
            return self.p * F.relu(x) + (1 - self.p) * F.relu(-x) * -1


class AID_GELU(nn.Module):
    def __init__(self, dropout_prob=0.9, approximate="none"):
        super().__init__()
        self.p = dropout_prob
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        # Apply GELU first
        gelu_output = self.gelu(x)

        if self.training:
            # Separate positive and negative components using masks
            pos_mask = (gelu_output > 0).float()
            neg_mask = (gelu_output <= 0).float()

            pos_vals = gelu_output * pos_mask
            neg_vals = gelu_output * neg_mask

            # Apply dropout with different probabilities
            pos_dropped = F.dropout(pos_vals, p=1 - self.p, training=True)
            if self.p > 0:
                pos_dropped = pos_dropped / self.p

            neg_dropped = F.dropout(neg_vals, p=self.p, training=True)
            if (1 - self.p) > 0:
                neg_dropped = neg_dropped / (1 - self.p)

            return pos_dropped + neg_dropped
        else:
            # Test time behavior
            pos_mask = (gelu_output > 0).float()
            neg_mask = (gelu_output <= 0).float()

            pos_vals = gelu_output * pos_mask
            neg_vals = gelu_output * neg_mask

            return self.p * pos_vals + (1 - self.p) * neg_vals
