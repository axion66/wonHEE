import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.positional_encoding import PositionalEncoding
from layers.rms import RMSNorm
from layers.utils import FFN

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
# from soft_moe_pytorch import SoftMoE
from st_moe_pytorch import MoE
from dataclasses import dataclass

'''
from gradnorm_pytorch import (
    GradNormLossWeighter,
    MockNetworkWithMultipleLosses
)
'''

MIN = 200


@dataclass
class MODEL_CONFIG():
    input_dim: int
    dim: int
    output_dim: int
    n_head: int
    seq_len: int
    use_rotary: bool
    use_moe: bool
    n_expert: int
    n_encoder_block: int
    n_sub_head_block: int
    n_class: int
    
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, use_rotary):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rotary = use_rotary

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim)
        else:
            self.rotary = None

    def forward(self, x, causal=False, attn_mask=None):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        if self.use_rotary:
            q, k = map(lambda t: self.rotary.rotate_queries_or_keys(t), (q, k))

        q = q / (self.head_dim ** 0.5)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        cfg: MODEL_CONFIG,
    ):
        super().__init__()
        self.norm_1 = RMSNorm(cfg.dim)
        self.norm_2 = RMSNorm(cfg.dim)
        self.self_attn = Attention(cfg.dim, cfg.n_head, use_rotary=cfg.use_rotary)
        self.ffn = MoE(dim=cfg.dim, num_experts=cfg.n_expert) if cfg.use_moe else FFN(dim=cfg.dim)
        self.use_moe = cfg.use_moe
    
    def forward(self, x, causal=False, attn_mask=None):
        
        # MoE + Attn
        if self.use_moe:
            x = x + self.ffn(self.norm_2(x))
            return x + self.self_attn(self.norm_1(x), causal=causal, attn_mask=attn_mask)
        
        
        # Attn + FFN
        x = x + self.self_attn(self.norm_1(x), causal=causal, attn_mask=attn_mask)
        return x + self.ffn(self.norm_2(x))


class TransformerHead(nn.Module):
    def __init__(
        self, 
        config: MODEL_CONFIG,
        output_dim: int,
        is_classification: bool = False,
    ):
        super().__init__()
        if is_classification:
            config.seq_len += 1 # for cls token
            
        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg=config,
            )
            for _ in range(config.n_sub_head_block)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim)) if is_classification else None
        self.out_proj = nn.Linear(config.dim, output_dim)

    def forward(self, x, attn_mask=None, causal=False):
        if self.cls_token is not None:
            x = torch.cat(
                [
                    self.cls_token.expand(x.size(0), -1, -1),
                    x
                ], dim=1
            ) 

        for block in self.blocks:
            x = block(x, causal=causal, attn_mask=attn_mask)

        if self.cls_token is not None:
            x = x[:, 0, :]  

        return self.out_proj(x)


class Transformer(nn.Module):
    def __init__(self, config: MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.encoder_embed = nn.Linear(config.input_dim, config.dim)
        self.missing_val_embedding = nn.Parameter(torch.randn(config.dim))
        self.positional_encoding = PositionalEncoding(config.dim, dropout=0.1, max_len=3000)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_encoder_block)
        ])

        self.classification_head = TransformerHead(config, output_dim=config.n_class, is_classification=True)
        self.forecasting_head = TransformerHead(config, output_dim=config.output_dim)
        self.imputation_head = TransformerHead(config, output_dim=config.output_dim)

    def encode(self, x, mask=None):
        x_embedded = self.encoder_embed(x)

        if mask is not None:
            x_embedded = torch.where(mask.unsqueeze(-1) == 1, x_embedded, self.missing_val_embedding)

        x = self.positional_encoding(x_embedded)
        for block in self.encoder_blocks:
            x = block(x, causal=False, attn_mask=None)
        return x

    def forward(self, inputs, mode='classification'):
        if isinstance(inputs, tuple):
            x, mask = inputs
        else:
            x, mask = inputs, None

        encoded_output = self.encode(x, mask)

        decoded = None
        if mode == 'imputation':
            decoded = self.imputation_head(encoded_output)
        elif mode == 'forecasting':
            decoded = self.forecasting_head(encoded_output, causal=True)
        elif mode == 'classification':
            return self.classification_head(encoded_output)
        else:
            raise ValueError("Invalid mode: choose 'classification', 'imputation', or 'forecasting'")
        
        return decoded

    def compute_loss(self, src, labels, mode, device):
        if mode == "classification":
            output = self((src, None), mode=mode)
            loss = F.cross_entropy(output, labels)
            stats = {
                "correct": (torch.argmax(output, dim=1) == labels).sum().item(),
                "total": labels.size(0)
            }
            return loss, stats, output, None

        seq_len = src.shape[1]
        
        if mode == "imputation":
            num_forecast = torch.randint(MIN, seq_len - (MIN // 2), (1,), device=device).item()
            forecast_idx = torch.randperm(seq_len, device=device)[:num_forecast]
            
            masked_src = src.clone()
            mask_tensor = torch.zeros(src.shape[0], src.shape[1], dtype=torch.bool, device=device)
            mask_tensor[:, forecast_idx] = True
            
            output = self((masked_src, mask_tensor), mode=mode)
            loss = F.mse_loss(output[:, forecast_idx, :], src[:, forecast_idx, :])
            # loss = F.mse_loss(output, src)
            full_output = output
            masked_indices = forecast_idx

        elif mode == "forecasting":
            num_forecast = torch.randint(MIN, seq_len - (MIN // 2), (1,), device=device).item()
            forecast_idx = torch.arange(seq_len - num_forecast, seq_len, device=device)
            
            masked_src = src.clone()
            mask_tensor = torch.zeros(src.shape[0], src.shape[1], dtype=torch.bool, device=device)
            mask_tensor[:, forecast_idx] = True
            
            output = self((masked_src, mask_tensor), mode=mode)
            loss = F.mse_loss(output[:, forecast_idx, :], src[:, forecast_idx, :])
            # loss = F.mse_loss(output, src)
            
            full_output = output
            masked_indices = forecast_idx
        else:
            raise ValueError("Invalid mode")

        return loss, None, full_output, masked_indices


if __name__ == "__main__":
    src = torch.randn((2, 50, 1))
    mask = torch.randint(0, 2, (2, 50))

    config = MODEL_CONFIG(
        input_dim=1,
        dim=64,
        output_dim=1,
        n_head=4,
        seq_len=50,
        use_rotary=False,
        use_moe=True,
        n_expert=4,
        n_encoder_block=2,
        n_sub_head_block=6,
        n_class=5
    )
    
    model = Transformer(config)

    out_cls = model((src, None), mode='classification')
    print("Classification output:", out_cls.shape)

    out_imp = model((src, mask), mode='imputation')
    print("Imputation output:", out_imp.shape)

    out_frc = model((src, mask), mode='forecasting')
    print("Forecasting output:", out_frc.shape)