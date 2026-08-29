import torch
import math
import torch.nn as nn

class SliceAwarePositionalEncoding(nn.Module):
    """
    Implements Slice-Aware Sinusoidal Positional Encoding (Definition 5.3).
    Includes fixed (Linear, Exponential, Harmonic) and Learnable frequency scaling strategies.
    """
    def __init__(self, s_max, ds, p, strategy="linear"):
        super().__init__()
        self.s_max = s_max
        self.ds = ds
        self.p = p
        self.strategy = strategy.lower()

        # Compute fixed alpha_k factors
        if self.strategy == "linear":
            alpha = torch.tensor([(k + 1) / p for k in range(p)])
        elif self.strategy == "exponential":
            if p > 1:
                alpha = torch.tensor([2 ** (k / (p - 1)) for k in range(p)])
            else:
                alpha = torch.tensor([1.0])
        elif self.strategy == "harmonic":
            alpha = torch.tensor([float(k + 1) for k in range(p)])
        elif self.strategy == "learnable":
            # Treat scaling factors as trainable parameters
            self.alpha = nn.Parameter(torch.ones(p))
        elif self.strategy == "standard":
            alpha = torch.ones(p)
        else:
            raise ValueError(f"Unknown frequency scaling strategy: {strategy}")

        if self.strategy != "learnable":
            self.register_buffer("alpha", alpha)

        # Compute base sinusoidal components
        pe = torch.zeros(s_max, ds, p)
        
        j = torch.arange(1, ds + 1).float()         # paper: j = 1..ds
        expo = 2.0 * torch.floor((j - 1) / 2.0) / ds
        div = 1/(10000.0 ** expo)
        self.register_buffer("div_term", div)
        self.register_buffer("pe_template", pe)

    def forward(self, batch_size, s, device):
        # We compute the slice-aware positional encoding dynamically up to seq length s
        pe = torch.zeros(s, self.ds, self.p, device=device, dtype=self.alpha.dtype)
        t = torch.arange(1,s+1, device=device, dtype=self.alpha.dtype).unsqueeze(1) # (s, 1)

        # Iterate over slices to apply alpha_k frequency scaling
        for k in range(self.p):
            scaled_t = t * self.alpha[k]
            # Even indices use sine, odd indices use cosine
            arg = scaled_t * self.div_term   # (s, 1) * (ds,) → (s, ds)
            pe[:, 0::2, k] = torch.sin(arg[:, 0::2])
            pe[:, 1::2, k] = torch.cos(arg[:, 1::2])
            

        # Expand to batch dimension: (B, s, ds, p)
        return pe.unsqueeze(0).expand(batch_size, -1, -1, -1)
