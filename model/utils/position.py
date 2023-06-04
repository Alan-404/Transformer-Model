import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __encode_positon(self, n_ctx: int) -> torch.Tensor:
        pos = torch.arange(n_ctx).unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> torch.Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1 / (torch.pow(10000, angles/embedding_dim))
        return angles.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.__encode_positon(x.size(1))
        angles = self.__encode_embedding(x.size(2))
        pos_angles = torch.matmul(pos, angles)

        pos_angles[0::2] = torch.sin(pos_angles[0::2])
        pos_angles[1::2] = torch.cos(pos_angles[1::2])

        x += pos_angles.unsqueeze(0).to(x.device)
        return x