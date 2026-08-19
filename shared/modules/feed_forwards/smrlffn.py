import torch.nn as nn
import torch
import math
from shared.tools.functions.dcttransform import DCTTransform

class SMRLFFN(nn.Module):
    """
    Implements L-Feed-Forward Network (Definition 5.5 / Theorem 5.6).
    Applies standard FFNs independently to each transform-domain slice.
    """
    def __init__(self, ds, d_ff_s, p, activation="relu"):
        super().__init__()
        self.ds = ds
        self.d_ff_s = d_ff_s
        self.p = p

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
        X_hat = DCTTransform.forward(X, Z)

        # 2. Reshape to treat slice mode as batch: (B, p, T, ds)
        X_hat_sliced = X_hat.permute(0, 3, 1, 2)

        # 3. Apply first linear layer: H = X_hat * W1 + b1
        # unsqueeze(0) transforms (p, 1, d_ff_s) to (1, p, 1, d_ff_s) to broadcast over Batch size B
        H = torch.einsum('b p t s, p s f -> b p t f', X_hat_sliced, self.W1) + self.b1.unsqueeze(0)

        # 4. Element-wise non-linearity
        G = self.act(H)

        # 5. Apply second linear layer: Y = G * W2 + b2
        # unsqueeze(0) transforms (p, 1, ds) to (1, p, 1, ds) to broadcast over Batch size B
        Y_hat_sliced = torch.einsum('b p t f, p f s -> b p t s', G, self.W2) + self.b2.unsqueeze(0)

        # 6. Permute back and transform to original domain
        Y_hat = Y_hat_sliced.permute(0, 2, 3, 1).contiguous()
        return DCTTransform.inverse(Y_hat, Z)