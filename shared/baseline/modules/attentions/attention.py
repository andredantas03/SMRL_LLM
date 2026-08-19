from torch import nn, Tensor
from einops import rearrange,einsum
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math

from shared.baseline.modules.positional_encoders.rope import RotaryEmbedding

class Causal_Multi_Head_Self_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout=0.0,
    ): 
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.W_Q = Parameter(torch.empty((d_model, d_model)))

        self.W_K = Parameter(torch.empty((d_model, d_model)))

        self.W_V = Parameter(torch.empty((d_model, d_model)))

        self.W_O = Parameter(torch.empty((d_model, d_model)))

        self.reset_parameters()
        self.rope = RotaryEmbedding(self.d_head)

    @torch.no_grad()
    def reset_parameters(self):
        std = 0.02
        for p in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            p_fp32 = torch.empty_like(p, dtype=torch.float32, device=p.device)
            nn.init.normal_(p_fp32, mean=0.0, std=std)
            p.copy_(p_fp32.to(p.dtype))
    
    def scaled_dot_product_attention(self, query, key, value, causal_mask = None):
      # bs, l, nh, dh = 
      *_, d_head = query.shape

      qk = einsum(query,key,
          "batch_size lq n_head d_head, batch_size lk n_head d_head -> batch_size n_head lq lk",
      )

      logits = qk / (d_head**0.5)

      if causal_mask is not None:
            logits = logits.masked_fill(causal_mask == 1, float('-inf'))

      attn_weights = F.softmax(logits, dim=-1)
      output = einsum(attn_weights, value,
          "batch_size n_head lq lk, batch_size lk n_head d_v -> batch_size n_head lq d_v",
      )

      # Return a tensor of shape "batch_size n_head lq d_v"
      return self.dropout(output)

    def forward(self, x):
        # Validando entrada e recolhendo dados
        *batch_dims, s, d_in = x.shape
        assert d_in == self.d_model 
        device = x.device
        causal_mask = torch.triu(torch.ones((s, s), device=device), 1).bool()

        # Realiza multiplicação unica
        # matmul_output = einsum(torch.cat((self.W_Q, self.W_K, self.W_V), dim=0),x,"n_head_x_d_head d_model, ... seq_len d_model ->... seq_len n_head_x_d_head")
        W_qkv = torch.cat((self.W_Q, self.W_K, self.W_V), dim=0)
        matmul_output = einsum(W_qkv, x, "nd d, ... s d -> ... s nd")

        # Separa as matrizes
        # [..., s, 3*h*d_k] -> [3, ..., n_head, s, d_head]
        qkv = rearrange(
            matmul_output,
            "... s (qkv n_head d_head) -> qkv ... s n_head d_head",
            qkv=3,
            n_head=self.n_head,
            d_head=self.d_head,
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # cada um: [..., s, n_head, d_head]

        # Rotaciona as matrizes Query e Key
        if self.rope is not None:
            query = self.rope(query)
            key = self.rope(key)

            
        # Executa a atenção
        attn_i = self.scaled_dot_product_attention(
            query=query, key=key, value=value, causal_mask=causal_mask  # [..., h, s, d_k]
        )  # -> [..., h, s, d_k]

        # Concatena a saida
        multihead_output = rearrange(attn_i, "... h s d_k -> ... s (h d_k)")
        output = einsum(
            self.W_O,
            multihead_output,
            "d_model num_heads_x_dk, ... seq_len num_heads_x_dk ->... seq_len d_model",
        )

        return output

