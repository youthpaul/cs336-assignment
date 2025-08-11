import torch
import torch.nn as nn
import math
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # truncated normal init: mean=0, std=sqrt(2/(in+out)), truncate at +/-3 std
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> output: (..., out_features)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        # weight shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # truncated normal init: mean=0, std=1, truncate at +/-3
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...,) long tensor
        return self.weight[token_ids]
















