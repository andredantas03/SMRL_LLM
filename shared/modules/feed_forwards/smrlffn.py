from einops import einsum
import torch.nn as nn
import torch
import math
from shared.tools.functions.l_product import l_product

class SMRLFFN(nn.Module):
    """
    Implements L-Feed-Forward Network (Definition 5.5 / Theorem 5.6).
    Applies standard FFNs independently to each transform-domain slice.
    """
    def __init__(self, ds, d_ff_s, p, activation="relu", dropout=0.1):
        super().__init__()
        self.ds = ds
        self.d_ff_s = d_ff_s
        self.p = p
        self.drop = nn.Dropout(dropout)

        # Transform-domain weight parameter tensors
        self.W1 = nn.Parameter(torch.empty(ds, d_ff_s, p))
        self.W2 = nn.Parameter(torch.empty(d_ff_s, ds, p))
        self.b1 = nn.Parameter(torch.empty(1, d_ff_s, p))
        self.b2 = nn.Parameter(torch.empty(1, ds, p))

        if activation == "relu":
            self.act = torch.relu
        elif activation == "gelu":
            self.act = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.reset_parameters()
    
    def reset_parameters(self):
        for W in (self.W1, self.W2):
            nn.init.kaiming_uniform_(W, nonlinearity='relu', mode='fan_in')
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, X, Z):
        # X: (B, T, ds, p); weights stored as (p, in, out) -> (in, out, p) for l_product
        H = l_product(X, self.W1, Z)+ self.b1
        G = self.act(H)
        output = l_product(G, self.W2, Z) + self.b2
        return output