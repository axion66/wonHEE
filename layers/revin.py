# Copied from https://github.com/ts-kim/RevIN/blob/master/RevIN.py
# reversible instance norm to remove bias
import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(
        self, 
        model_dim: int,
        eps=1e-5, 
        affine=True
    ):
        """
        :param model_dim: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.model_dim = model_dim
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.model_dim))
        self.affine_bias = nn.Parameter(torch.zeros(self.model_dim))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.model_dim}, eps={self.eps}, affine={self.affine})')


if __name__ == "__main__":
    revin = RevIN(
        model_dim=64, 
        eps=1e-5,
        affine=True
    )
    
    x = torch.randn((16, 300, 64)) # [batch, length, dim]    
    normalized_x = revin(x, mode="norm")
    denormalized_x = revin(x, mode="denorm")
    
    
    print(f"x.shape: {x.shape}")
    print(f"normalized_x.shape: {normalized_x.shape}")
    print(f"denormalized_x.shape: {denormalized_x.shape}")
    
    '''
        x.shape: torch.Size([16, 300, 64])
        normalized_x.shape: torch.Size([16, 300, 64])
        denormalized_x.shape: torch.Size([16, 300, 64])
    '''