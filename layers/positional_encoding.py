# Borrowed and Modified from https://github.com/MediaBrain-SJTU/CoFormer/blob/main/models/model_medical
# Three choices: 
# Use regular PE -> Not possible as it expects equal-distance.
# Use padding -> simple but model might understand padding as a value.
# use distance-aware encoding -> like PositionalEncoding_irregular_plus.
# or RNN or Neural-ODE, but we love transformer.
# Dropout: https://discuss.pytorch.org/t/why-use-dropout-in-positional-encoding-layer/159923

import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim, # 64, 128, 512
        dropout, # 0.0, 0.2, 0.3
        max_len): # 300, 500, 1000
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0)) 

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding_irregular(nn.Module):
    
    def __init__(
        self, 
        model_dim, 
        dropout, 
    ):
        
        super().__init__()
        self.model_dim = model_dim
        self.dropout = nn.Dropout(p=dropout)
         
    def forward(self, x, time):
        '''
            Expects varying-length input. 
            so in forward(self, x, time) x and time can have varying input length (varying max_len PE, i would say)
        '''
        pe = torch.zeros(x.shape[0],time.shape[1], self.model_dim).to(x.device)
        div = torch.exp(torch.arange(0., self.model_dim, 2) * -(math.log(10000.0) / self.model_dim)).to(x.device)
        
        pe[:,:, 0::2] = torch.sin(time * div)   
        pe[:,:, 1::2] = torch.cos(time * div)  

        x = x + torch.autograd.Variable(pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionalEncoding_irregular_trainable(nn.Module):
    def __init__(
        self, 
        model_dim, 
        dropout,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.model_dim = model_dim
        self.linear = nn.Linear(2 * model_dim, model_dim)
         
    def forward(self, x, time):
        '''
            self.linear(concat(x + position))
            
            
            
            x = {0.3, 0.1, 0.5, 0.1, 0.2} - irregular
            t = {0, 1, 3, 11, 19}
        '''
        pe = torch.zeros(x.shape[0],time.shape[1], self.model_dim).to(x.device)
        div = torch.exp(torch.arange(0., self.model_dim, 2) * -(math.log(10000.0) / self.model_dim)).to(x.device)
         
        pe[:,:, 0::2] = torch.sin(time * div)
        pe[:,:, 1::2] = torch.cos(time * div) 

        fuse = torch.cat((x,torch.autograd.Variable(pe[:, :x.size(1)], requires_grad=False)),dim=-1)
        x = self.linear(fuse)
        return self.dropout(x)



# Other embeddings later can be found in: https://github.com/thuml/TimeXer/blob/main/layers/Embed.py
    
    
if __name__ == "__main__":
    x = torch.randn((16, 300, 64)) # [batch, length, dim]
    t = torch.arange(0, 300, 1)
    t = t.unsqueeze(0).expand(x.size(0), -1) .unsqueeze(-1)
    
    print(f"x : {x.shape}")
    print(f"t : {t.shape}")
    '''
        x : torch.Size([16, 300, 64])
        t : torch.Size([16, 300, 1])
    '''
    
    model_dim = 64
    dropout = 0.1
    max_len = 300
    
    pe = PositionalEncoding(
        model_dim, dropout, max_len
    )
    
    pe_irregular = PositionalEncoding_irregular(
        model_dim, dropout
    )
    
    pe_irregular_trainable = PositionalEncoding_irregular_trainable(
        model_dim, dropout
    )
    
    
    out_pe = pe(x)
    out_pe_irregular = pe_irregular(x, t)
    out_pe_irregular_trainable = pe_irregular_trainable(x, t)
    
    print(out_pe.shape)
    print(out_pe_irregular.shape)
    print(out_pe_irregular_trainable.shape)
    '''
        torch.Size([16, 300, 16])
        torch.Size([16, 300, 16])
        torch.Size([16, 300, 16])
    '''
    
    
    