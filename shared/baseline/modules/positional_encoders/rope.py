from torch import nn
import torch


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.max_seq_len_cached = 0
        self.d_head = d_head
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_head, 2).float().to(device) / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )

            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", emb.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin().to(dtype), persistent=False
            ) # seq_len, dim

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    
    def forward(self, x):
        bs, l, nh, dh = x.shape
        self._set_cos_sin_cache(seq_len=l, device=x.device, dtype=x.dtype)
        cos = self.cos_cached[:l].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:l].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)