import torch
import torch.nn as nn
from typing import Union, Callable
from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForwardNetworks
from .residual import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        # Main layers
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForwardNetworks(d_ff=d_ff, d_model=d_model, activation=activation)

        # Residual Connections
        self.residual_1 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_2 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        attention, _ = self.attention(x, x, x, mask)
        attention = self.residual_1(attention, x)

        # sublayer 2
        ffn_output = self.ffn(attention)
        output = self.residual_2(ffn_output, attention)

        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        # Main layers
        self.local_attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.global_attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForwardNetworks(d_ff=d_ff, d_model=d_model, activation=activation)

        # Residual Connections
        self.residual_1 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_2 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_3 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: Union[torch.Tensor, None], padding_mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        local_attention, _ = self.local_attention(x, x, x, look_ahead_mask)
        local_attention = self.residual_1(local_attention, x)
        
        # sublayer 2
        global_attention, _ = self.global_attention(local_attention, encoder_output, encoder_output, padding_mask)
        global_attention = self.residual_2(global_attention, local_attention)

        # sublayer 3
        ffn_output = self.ffn(global_attention)
        output = self.residual_3(ffn_output, global_attention)

        return output