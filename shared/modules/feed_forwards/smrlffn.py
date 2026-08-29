from einops import einsum
import torch.nn as nn
import torch
import math
from shared.tools.functions import l_product
from shared.tools.functions.orthogonaltransform import OrthogonalTransform

class SMRLFFN(nn.Module):
    """
    Implements L-Feed-Forward Network (Definition 5.5 / Theorem 5.6).
    Applies standard FFNs independently to each transform-domain slice.
    """
    def __init__(self, ds, d_ff_s, p, activation="gelu", dropout=0.1):
        super().__init__()
        self.ds = ds
        self.d_ff_s = d_ff_s
        self.p = p
        self.drop = nn.Dropout(dropout)

        # Transform-domain weight parameter tensors
        self.W1 = nn.Parameter(torch.empty(p, ds, d_ff_s))
        self.W2 = nn.Parameter(torch.empty(p, d_ff_s, ds))
        self.b1 = nn.Parameter(torch.empty(p, 1, d_ff_s))
        self.b2 = nn.Parameter(torch.empty(p, 1, ds))

        if activation == "relu":
            self.act = torch.relu
        elif activation == "gelu":
            self.act = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.reset_parameters()

    def reset_parameters(self):
        for W in [self.W1, self.W2]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, X, Z):
        # 1. Transform activations to spectral domain: (B, T, ds, p)
        H = l_product(X,self.W1,Z)

        # 4. Element-wise non-linearity
        G = self.act(H)
        G = self.drop(G)
        # 5. Apply second linear layer: Y = G * W2 + b2
        # unsqueeze(0) transforms (p, 1, ds) to (1, p, 1, ds) to broadcast over Batch size B
        Y = l_product(G,self.W2,Z) + self.b2.unsqueeze(0)        
        return Y