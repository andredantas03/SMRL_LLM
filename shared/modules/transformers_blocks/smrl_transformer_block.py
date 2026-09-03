from shared.modules.attentions.attention import SMRL_Attention
from shared.modules.feed_forwards.smrlffn import SMRLFFN
from shared.modules.norms.smrl_layernorm import SMRL_LayerNorm
import torch.nn as nn

class SMRLTransformerBlock(nn.Module):
    """
    Implements a single Tensor Transformer Encoder Layer (Definition 5.8).
    Supports Post-LayerNorm (as in paper equations) or standard Pre-LayerNorm.
    """
    def __init__(self, d, p, h, d_ff,activation="relu", norm_first=False, eps=1e-5, causal=False, dropout=0.1):
        super().__init__()
        self.ds = d // p
        self.p = p

        self.drop = nn.Dropout(dropout)
        self.mhal_layer = SMRL_Attention(self.ds, p, h, causal=causal, dropout=dropout)
        self.ffn_layer = SMRLFFN(self.ds, self.ds * 4, p, activation=activation, dropout=dropout)
        self.tln1 = SMRL_LayerNorm(self.ds, p, eps=eps)
        self.tln2 = SMRL_LayerNorm(self.ds, p, eps=eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, X, P, Z, attention_mask=None):
        # Post-LN variant (Definition 5.8)
        X_pos = X + P if P is not None else X
        # Equation 5.3: X' = TLN( X + MHAL(X_pos) )
        X_prime = self.tln1(X_pos + self.drop1(self.mhal_layer(X_pos, Z, attention_mask=attention_mask)))
        
        # Equation 5.4: Y = TLN( X' + TFFN(X') )
        Y = self.tln2(X_prime + self.drop2(self.ffn_layer(X_prime, Z)))
        return Y