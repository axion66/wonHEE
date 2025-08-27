import torch
import torch.nn as nn
import math

from layers.activation import get_activation

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"
    

    
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FFN(nn.Module):
    def __init__(
        self, 
        dim,
        expand_ratio = 4,
        dropout = 0.2,
        activation = 'swiglu',
    ):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand_ratio),
            get_activation(activation, dim * expand_ratio),
            nn.Dropout(p=dropout),
            nn.Linear(dim * expand_ratio, dim)
        )
        
    def forward(self, x):
        return self.ffn(x)