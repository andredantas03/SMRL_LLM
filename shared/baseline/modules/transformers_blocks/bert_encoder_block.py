import torch.nn as nn
from torch import Tensor

from ..feed_forwards.feedforward import PositionwiseFFN
from ..attentions.bidirectional_attention import Bidirectional_Multi_Head_Self_Attention


class BertEncoder_Block(nn.Module):
    """Vanilla Transformer encoder layer, Post-LN (paper Std)."""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1, activation="gelu", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = dropout

        self.att = Bidirectional_Multi_Head_Self_Attention(
            d_model=self.d_model,
            n_head=self.n_head,
            dropout=self.dropout,
            use_rope=False,
        )
        self.ffn = PositionwiseFFN(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=activation,
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

    def forward(self, x: Tensor, attention_mask=None):
        x = self.norm1(x + self.drop1(self.att(x, attention_mask=attention_mask)))
        return self.norm2(x + self.drop2(self.ffn(x)))
