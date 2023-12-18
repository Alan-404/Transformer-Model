import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_samples = d_model // heads

        self.sqrt_sample = math.sqrt(self.head_samples)

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> [torch.Tensor, torch.Tensor]:
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/self.sqrt_sample

        if mask is not None:
            attention_scores += mask * (float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_context = torch.matmul(attention_weights, v)
        
        return attention_context, attention_weights
    
    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, _ = x.size()

        x = torch.reshape(x, (batch_size, n_ctx, self.heads, self.head_samples))
        x = torch.permute(x, (0, 2, 1, 3))
        
        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> [torch.Tensor, torch.Tensor]:
        self_attention = (self.heads is None or self.heads == 1)

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        if self_attention == False:
            qw = self.split_head(qw)
            kw = self.split_head(kw)
            vw = self.split_head(vw)
        
        attention_context, attention_weights = self.scaled_dot_product_attention(qw, kw, vw, mask)

        if self_attention == False:
            attention_context = torch.permute(attention_context, (0, 2, 1, 3))
            attention_context = torch.reshape(attention_context, (q.size(0), q.size(1), self.d_model))
        
        attention_context = self.linear_output(attention_context)

        return attention_context, attention_weights