import math
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

        # Trainable parameters stored directly in transform domain
        # Shapes: (p, ds, ds) for projection matrices within each slice
        self.attn_drop = nn.Dropout(dropout)
        self.W_q = nn.Parameter(torch.empty(p, ds, ds))
        self.W_k = nn.Parameter(torch.empty(p, ds, ds))
        self.W_v = nn.Parameter(torch.empty(p, ds, ds))
        self.W_o = nn.Parameter(torch.empty(p, ds, ds))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for projections
        for W in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(W, mean=0.0, std=0.02)

    def forward(self, X_pos, Z, attention_mask=None):
        # 1. Transform input activations to DCT domain: (B, s, ds, p)
        X_hat = OrthogonalTransform.forward(X_pos, Z)

        # 2. Treat slice index as a batch dimension for GPU concurrency:
        # Permute (B, s, ds, p) -> (B, p, s, ds)
        X_hat_sliced = rearrange(X_hat, "b s ds p -> b p s ds")
        
        # 3. Project queries, keys, and values within each slice
        # X_hat_sliced is (B, p, s, ds), self.W_* is (p, ds, ds)
        Q = einsum(X_hat_sliced, self.W_q, "b p s din, p din dout -> b p s dout")
        K = einsum(X_hat_sliced, self.W_k, "b p s din, p din dout -> b p s dout")
        V = einsum(X_hat_sliced, self.W_v, "b p s din, p din dout -> b p s dout")

        # 4. Split into h attention heads: (B, p, s, h, d_h) -> Transpose to (B, p, h, s, d_h)
        Q = rearrange(Q, "b p s (h dh) -> b p h s dh", h=self.h, dh=self.dh)
        K = rearrange(K, "b p s (h dh) -> b p h s dh", h=self.h, dh=self.dh)
        V = rearrange(V, "b p s (h dh) -> b p h s dh", h=self.h, dh=self.dh)

        # 5. Fuse slice and head dimensions to run standard batched attention
        # Shape: (B, p * h, s, d_h)
        Q_fused = rearrange(Q, "b p h s dh -> b (p h) s dh")
        K_fused = rearrange(K, "b p h s dh -> b (p h) s dh")
        V_fused = rearrange(V, "b p h s dh -> b (p h) s dh")

        # 6. Scaled Dot-Product Attention inside slices
        scores = einsum(Q_fused, K_fused, "b f sq dh, b f sk dh -> b f sq sk") / math.sqrt(self.dh) # (B, p * h, s, s)
        
        s = scores.size(-1)
        block = torch.zeros(s, s, dtype=torch.bool, device=scores.device)
        if self.causal:
            block = torch.triu(torch.ones(s, s, dtype=torch.bool, device=scores.device), 1)
        if attention_mask is not None:
            # (B, s) -> (B, 1, 1, s) no eixo das keys
            pad = (attention_mask == 0)[:, None, None, :]
            block = block | pad
        scores = scores.masked_fill(block, float("-inf"))

        
        attn_weights = torch.softmax(scores, dim=-1)
        context = einsum(attn_weights, V_fused, "b f sq sk, b f sk dh -> b f sq dh") # (B, p * h, s, dh)
        context = self.attn_drop(context)

        # 7. Unfuse dimensions and concatenate heads
        # Shape: (B, (p*h), s, dh) -> (B, p, s, h, dh) -> (B, p, s, ds)
        context = rearrange(context, "b (p h) s dh -> b p s h dh", p=self.p, h=self.h)
        H = rearrange(context, "b p s h dh -> b p s (h dh)")

        # 8. Output projection within slice: (B, p, s, ds)
        Y_hat_sliced = einsum(H, self.W_o, "b p s din, p din dout -> b p s dout")

        # 9. Form Y_hat by permuting back to frontal slices: (B, s, ds, p)
        Y_hat = rearrange(Y_hat_sliced, "b p s ds -> b s ds p")

        # 10. Map back using inverse L-transform
        Y = OrthogonalTransform.inverse(Y_hat, Z)
        return Y


