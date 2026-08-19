import torch.nn as nn
from einops import rearrange,einsum
from torch import Tensor

from ..norms.rmsnorm import RMSNorm
from ..feed_forwards.feedforward import SwiGLU
from ..attentions.attention import (
  Causal_Multi_Head_Self_Attention,
  )

class Transformer_Block(nn.Module):
    def __init__(self,d_model,n_head,d_ff,**kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = kwargs.get("dropout")


        self.att = Causal_Multi_Head_Self_Attention(            
                d_model= self.d_model,
                n_head = self.n_head,
                dropout = self.dropout,
            )
        
        self.rmsnorm1 = RMSNorm(d_model=self.d_model)
        self.rmsnorm2 = RMSNorm(d_model=self.d_model)
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)       
        self.pwffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, X: Tensor):
        h = self.att(self.rmsnorm1(X))
        h = X + self.drop1(h)         
        return  h + self.drop2(self.pwffn(self.rmsnorm2(h)))