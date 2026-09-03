import math
from shared.tools.functions.l_product import l_transform, l_transform_inverse
from shared.tools.functions.orthogonaltransform import OrthogonalTransform
from torch import nn, Tensor
from einops import einsum, rearrange
import torch
import numpy as np


class SMRL_Attention(nn.Module):
    """
    Implements L-Multi-Head Attention (Algorithm 1 and 2).
    Optimized to treat the slice index as a batch dimension (Parallel implementation).
    """
    def __init__(self, ds, p, h, causal=False, dropout=0.1):
        super().__init__()
        self.ds = ds
        self.p = p
        self.h = h
        assert ds % h == 0, f"Slice width ds={ds} must be divisible by heads h={h}"
        self.dh = ds // h
        self.causal = causal

        self.attn_drop = nn.Dropout(dropout)
        self.W_q = nn.Parameter(torch.empty(h, ds, self.dh, p))
        self.W_k = nn.Parameter(torch.empty(h, ds, self.dh, p))
        self.W_v = nn.Parameter(torch.empty(h, ds, self.dh, p))
        self.W_o = nn.Parameter(torch.empty(ds, ds, p))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for projections
        for W in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.kaiming_uniform_(W)

    def forward(self, X_pos, Z, attention_mask=None):
        # 1. Transform input activations to DCT domain: (B, s, ds, p)
        X_hat = l_transform(X_pos, Z)

        
        
        # 3. Project queries, keys, and values within each slice
        # X_hat_sliced is (B, p, s, ds), self.W_* is (p, ds, ds)
        Q = einsum(X_hat, self.W_q, "b s ds p, h ds dh p-> b p s h dh")
        K = einsum(X_hat, self.W_k, "b s ds p, h ds dh p-> b p s h dh")
        V = einsum(X_hat, self.W_v, "b s ds p, h ds dh p-> b p s h dh")
        

        # 6. Scaled Dot-Product Attention inside slices
        scores = einsum(Q, K, "b p sq h dh, b p sk h dh -> b p h sq sk") / math.sqrt(self.dh) 
        
        s = scores.size(-1)
        block = torch.zeros(s, s, dtype=torch.bool, device=scores.device)
        if self.causal:
            block = torch.triu(torch.ones(s, s, dtype=torch.bool, device=scores.device), 1)
        if attention_mask is not None:
            # (B, s) -> (B, 1, 1, s) no eixo das keys
            pad = (attention_mask == 0)[:, None,None, None, :]
            block = block | pad
        scores = scores.masked_fill(block, float("-inf"))

        
        attn_weights = torch.softmax(scores, dim=-1)
        context = einsum(attn_weights, V, "b p h sq sk, b p sk h dh -> b p h sq dh")
        context = self.attn_drop(context)

        # 7. Unfuse dimensions and concatenate heads
        # Shape: (B, (p*h), s, dh) -> (B, p, s, h, dh) -> (B, p, s, ds)
        context = rearrange(context, "b p h s dh -> b p s h dh", p=self.p, h=self.h)
        H = rearrange(context, "b p s h dh -> b p s (h dh)")

        # 8. Output projection within slice: (B, p, s, ds)
        Y_hat_sliced = einsum(H, self.W_o, "b p s din, din dout p-> b p s dout")

        # 9. Form Y_hat by permuting back to frontal slices: (B, s, ds, p)
        Y_hat = rearrange(Y_hat_sliced, "b p s ds -> b s ds p")

        # 10. Map back using inverse L-transform
        Y = l_transform_inverse(Y_hat, Z)
        return Y


