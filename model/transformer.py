import torch
import torch.nn as nn
from typing import Union, Callable
from .components.encoder import Encoder
from .components.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                encoder_token_size: int,
                decoder_token_size: int,
                n: int,
                d_model: int,
                heads: int,
                d_ff: int,
                dropout_rate: float,
                eps: float,
                activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.encoder = Encoder(
            token_size=encoder_token_size,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation
        )
        self.decoder = Decoder(
            token_size=decoder_token_size,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation
        )

    def generate_look_ahead_mask(self, padding_mask: torch.Tensor):
        pass

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, padding_mask: Union[torch.Tensor, None], look_ahead_mask: Union[torch.Tensor, None]) -> torch.Tensor:
        encoder_output = self.encoder(encoder_input, padding_mask)

        decoder_output = self.decoder(decoder_input, encoder_output, look_ahead_mask, padding_mask)

        return decoder_output