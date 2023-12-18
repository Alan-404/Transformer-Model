import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        angles = self.__encode_embedding(d_model).unsqueeze(0)
        self.register_buffer('angles', angles)

    def __encode_positon(self, n_ctx: int) -> torch.Tensor:
        pos = torch.arange(n_ctx).unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> torch.Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1 / (torch.pow(10000, angles/embedding_dim))
        return angles.unsqueeze(0)
    
    def forward(self, n_ctx: int) -> torch.Tensor:
        pos = self.__encode_positon(n_ctx).to(self.angles.device)

        pos_angles = torch.matmul(pos, self.angles)

        pos_angles[:, 0::2] = torch.sin(pos_angles[:, 0::2])
        pos_angles[:, 1::2] = torch.cos(pos_angles[:, 1::2])

        return pos_angles.unsqueeze(0)
