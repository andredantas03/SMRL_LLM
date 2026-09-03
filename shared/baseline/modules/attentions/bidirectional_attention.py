from torch import nn
from einops import rearrange, einsum
import torch
import torch.nn.functional as F


class Bidirectional_Multi_Head_Self_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout=0.1,
        use_rope=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, f"d_model={d_model} must be divisible by n_head={n_head}"
        self.d_head = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.W_Q = nn.Linear(d_model, d_model,bias=True)
        self.W_K = nn.Linear(d_model, d_model,bias=True)
        self.W_V = nn.Linear(d_model, d_model,bias=True)
        self.W_O = nn.Linear(d_model, d_model,bias=True)

        

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None):
        *_, d_head = query.shape

        qk = einsum(
            query,
            key,
            "batch_size lq n_head d_head, batch_size lk n_head d_head -> batch_size n_head lq lk",
        )
        logits = qk / (d_head**0.5)

        if attn_mask is not None:
            key_pad = (attn_mask == 0)[:, None, None, :]
            logits = logits.masked_fill(key_pad, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = einsum(
            attn_weights,
            value,
            "batch_size n_head lq lk, batch_size lk n_head d_v -> batch_size n_head lq d_v",
        )
        return output

    def forward(self, x, attention_mask=None):
        assert x.shape[-1] == self.d_model

        query = rearrange(
            self.W_Q(x), "... s (h d) -> ... s h d", h=self.n_head, d=self.d_head
        )
        key = rearrange(
            self.W_K(x), "... s (h d) -> ... s h d", h=self.n_head, d=self.d_head
        )
        value = rearrange(
            self.W_V(x), "... s (h d) -> ... s h d", h=self.n_head, d=self.d_head
        )

        attn_out = self.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
        )

        multihead_output = rearrange(attn_out, "... h s d -> ... s (h d)")
        return self.W_O(multihead_output)
