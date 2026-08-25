import torch.nn as nn
from einops import rearrange,einsum
from torch import Tensor
import torch
from torch.nn import Parameter
import math

class PositionwiseFFN(nn.Module):
    """Standard Transformer FFN: Linear → GELU → Linear, width 4d unless overridden."""

    def __init__(self, d_model, d_ff=None, dropout=0.0, activation="gelu"):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        if activation == "relu":
            self.act = torch.relu
        elif activation == "gelu":
            self.act = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d_model = d_model

        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight1 = Parameter(
            torch.empty((self.d_ff, self.d_model), **factory_kwargs)
        )

        self.weight2 = Parameter(
            torch.empty((self.d_model, self.d_ff), **factory_kwargs)
        )

        self.weight3 = Parameter(
            torch.empty((self.d_ff, self.d_model), **factory_kwargs)
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        std = 0.02
        for p in (self.weight1, self.weight2, self.weight3):
            tmp = torch.empty_like(p, dtype=torch.float32, device=p.device)
            nn.init.normal_(tmp, mean=0.0, std=std)
            p.copy_(tmp.to(p.dtype))
        # Ajuste opcional:
        self.weight2.mul_(1.0 / math.sqrt(2.0))

    def forward(self, x: Tensor):

        x_silu = einsum(self.weight1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x_1 = nn.functional.silu(x_silu)
        x_3 = einsum(self.weight3, x, "d_ff d_model, ... d_model -> ... d_ff")
        x = einsum(x_3, x_1, "... d_ff, ... d_ff -> ... d_ff")

        return einsum(self.weight2, x, "d_model d_ff, ... d_ff -> ... d_model")

