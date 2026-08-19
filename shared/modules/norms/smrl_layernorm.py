import torch.nn as nn
import torch

class SMRL_LayerNorm(nn.Module):
    """
    Implements Tensor Layer Normalization (Definition 5.7).
    Computes mean and variance along feature mode-2 slice-by-slice.
    """
    def __init__(self, d_s, p, eps=1e-5):
        super().__init__()
        self.d_s = d_s
        self.p = p
        self.eps = eps

        # Gamma and Beta parameters are defined slice-by-slice
        self.gamma = nn.Parameter(torch.ones(1, 1, d_s, p))
        self.beta = nn.Parameter(torch.zeros(1, 1, d_s, p))

    def forward(self, X):
        # X shape: (B, T, d_s, p)
        mean = X.mean(dim=2, keepdim=True) # mean along mode-2
        var = X.var(dim=2, keepdim=True, unbiased=False) # variance along mode-2
        
        X_norm = (X - mean) / torch.sqrt(var + self.eps)
        return self.gamma * X_norm + self.beta
