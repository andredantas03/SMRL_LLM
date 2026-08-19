from torch import nn
from torch.nn import Parameter
import torch


class CP_Embedding(nn.Module):
    def __init__(self, vocab_size, factors, d_cp, rank, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.factors = factors
        self.d_cp = d_cp
        self.rank = rank
        self.weight = Parameter(torch.empty((self.vocab_size, self.factors, self.d_cp, self.rank)))
        self.reset_parameters_orthogonal()

    @torch.no_grad()
    def reset_parameters(self):
        std_init = 0.02
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std_init, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


    @torch.no_grad()
    def reset_parameters_orthogonal(self):
        if self.d_cp < self.rank:
            raise ValueError(
                f"Ortogonalidade por modo exige d_cp >= rank "
                f"(got d_cp={self.d_cp}, rank={self.rank})"
            )
        std_init = 0.02
        # (V, f, d_cp, r) -> amostra Gaussiana, depois QR por (v, f)
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        V, F, D, R = self.weight.shape
        w = self.weight.reshape(V * F, D, R)  # um modo por vez
        # QR: Q tem colunas ortonormais em R^D
        Q, _ = torch.linalg.qr(w, mode="reduced")  # (V*F, D, R)
        # escala similar à trunc_normal_ antiga
        self.weight.copy_((Q * (std_init * (D ** 0.5))).reshape(V, F, D, R))