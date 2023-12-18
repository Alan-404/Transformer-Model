import torch
import torch.nn as nn
from typing import Callable, Optional
from ..utils.layer import DecoderLayer
from ..utils.position import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.postional_encoding = PositionalEncoding()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])
        self.classifier = nn.Linear(in_features=d_model, out_features=token_size)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        x += self.postional_encoding(x.size(1))
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)
        x = self.classifier(x)
        return x