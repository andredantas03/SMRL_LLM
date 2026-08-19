from torch import nn
from torch.nn import Parameter
import torch


class CP_Sparse_Embedding(nn.Module):
    def __init__(self, vocab_size, factors, d_cp, rank, n_dict_features, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.factors = factors
        self.d_cp = d_cp
        self.rank = rank
        self.dict_features = Parameter(torch.empty((n_dict_features, self.factors, self.d_cp, 1)))
        self.weight = Parameter(torch.empty((self.vocab_size, self.factors, self.d_cp, self.rank)))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        std_init = 0.02
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std_init, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
