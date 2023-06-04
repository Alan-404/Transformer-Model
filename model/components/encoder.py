import torch
import torch.nn as nn
from typing import Callable, Union
from ..utils.layer import EncoderLayer
from ..utils.position import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.postional_encoding = PositionalEncoding()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])
    
    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        x = self.embedding(x)
        x = self.postional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x