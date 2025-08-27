import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return nn.functional.normalize(x, dim = -1) * self.gamma * self.scale
