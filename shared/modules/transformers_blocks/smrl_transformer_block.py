from shared.modules.attentions.attention import SMRL_Attention
from shared.modules.feed_forwards.smrlffn import SMRLFFN
from shared.modules.norms.smrl_layernorm import SMRL_LayerNorm
import torch.nn as nn

class SMRLTransformerBlock(nn.Module):
    """
    Implements a single Tensor Transformer Encoder Layer (Definition 5.8).
    Supports Post-LayerNorm (as in paper equations) or standard Pre-LayerNorm.
    """
    def __init__(self, d, p, h, d_ff, activation="relu", eps=1e-5, norm_first=False, causal=False):
        super().__init__()
        self.ds = d // p
        self.p = p
        self.norm_first = norm_first

        self.mhal_layer = SMRL_Attention(self.ds, p, h, causal=causal)
        self.ffn_layer = SMRLFFN(self.ds, self.ds * 4, p, activation=activation)
        self.tln1 = SMRL_LayerNorm(self.ds, p, eps=eps)
        self.tln2 = SMRL_LayerNorm(self.ds, p, eps=eps)

    def forward(self, X, P, Z):
        if self.norm_first:
            # Pre-LN variant
            X_norm = self.tln1(X)
            X_pos = X_norm + P if P is not None else X_norm
            X = X + self.mhal_layer(X_pos, Z)
            
            X_norm2 = self.tln2(X)
            X = X + self.ffn_layer(X_norm2, Z)
            return X
        else:
            # Post-LN variant (Definition 5.8)
            X_pos = X + P if P is not None else X
            # Equation 5.3: X' = TLN( X + MHAL(X_pos) )
            X_prime = self.tln1(X + self.mhal_layer(X_pos, Z))
            
            # Equation 5.4: Y = TLN( X' + TFFN(X') )
            Y = self.tln2(X_prime + self.ffn_layer(X_prime, Z))
            return Y