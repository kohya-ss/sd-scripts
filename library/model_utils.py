from torch import nn
import torch.nn.functional as F
import torch


class AID(nn.Module):
    def __init__(self, dropout_prob=0.9):
        super(AID, self).__init__()
        self.p = dropout_prob
        self.training = True
    
    def forward(self, x):
        if self.training:
            # Generate masks for positive and negative values
            pos_mask = torch.bernoulli(torch.full_like(x, self.p))
            neg_mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            
            # Apply masks to positive and negative parts
            pos_part = F.relu(x) * pos_mask
            neg_part = F.relu(-x) * neg_mask * -1
            
            return pos_part + neg_part
        else:
            # During testing, use modified leaky ReLU with coefficient p
            return self.p * F.relu(x) + (1 - self.p) * F.relu(-x) * -1

