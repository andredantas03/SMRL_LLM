import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=128,
        dilation=1,
        dropout=0.1,
        attention_mode="sliding_window"
    ):
        """
        Longformer Multi-Head Self-Attention.
        
        This implementation supports:
        1. Sliding Window Attention (local context)
        2. Dilated Sliding Window Attention (extended local context)
        3. Global Attention (symmetric attention at specific pre-selected locations)
        4. Separate linear projections for sliding window and global attention
        
        Args:
            dim (int): Input embedding dimension (d_m)
            num_heads (int): Number of attention heads (h)
            window_size (int): Receptive field size (w). Each token attends to w/2 tokens on each side.
            dilation (int or list): Dilation factor (d). Can be a single integer or a list of length num_heads.
            dropout (float): Dropout probability for attention weights
            attention_mode (str): Only "sliding_window" is supported for this standard PyTorch 
                                 implementation which utilizes masks.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        
        # Dilation per head
        if isinstance(dilation, int):
            self.dilations = [dilation] * num_heads
        elif isinstance(dilation, (list, tuple)):
            assert len(dilation) == num_heads, f"Dilation list length must match num_heads {num_heads}"
            self.dilations = list(dilation)
        else:
            raise TypeError("Dilation must be an integer or a list/tuple of integers")
            
        # Linear projections for sliding window attention (local)
        self.q_proj_s = nn.Linear(dim, dim, bias=False)
        self.k_proj_s = nn.Linear(dim, dim, bias=False)
        self.v_proj_s = nn.Linear(dim, dim, bias=False)
        
        # Linear projections for global attention
        self.q_proj_g = nn.Linear(dim, dim, bias=False)
        self.k_proj_g = nn.Linear(dim, dim, bias=False)
        self.v_proj_g = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize global projection weights with local projection weights
        self._init_global_projections()
        
    def _init_global_projections(self):
        """Initializes global projections with sliding window projection weights as in the paper."""
        with torch.no_grad():
            self.q_proj_g.weight.copy_(self.q_proj_s.weight)
            self.k_proj_g.weight.copy_(self.k_proj_s.weight)
            self.v_proj_g.weight.copy_(self.v_proj_s.weight)

    def _create_attention_mask(self, seq_len, device, global_attention_mask=None):
        """
        Creates a custom 4D attention mask of shape (B, h, seq_len, seq_len)
        representing the Longformer attention patterns:
        - Sliding window attention (local w/2 elements on each side)
        - Dilated sliding window attention
        - Global attention (symmetric: attending to and from global locations)
        """
        # Base sliding window mask for each head depending on its dilation
        # Create coordinates
        grid = torch.arange(seq_len, device=device)
        # Distances: grid_row - grid_col
        dist = grid.unsqueeze(1) - grid.unsqueeze(0) # (seq_len, seq_len)
        
        masks = []
        for h in range(self.num_heads):
            d = self.dilations[h]
            w = self.window_size
            half_w = w // 2
            
            # Sliding window constraint: |i - j| <= (w / 2) * d
            # Dilation constraint: (i - j) must be divisible by d
            if d == 1:
                # Standard sliding window
                mask_h = (dist.abs() <= half_w)
            else:
                # Dilated sliding window
                mask_h = (dist.abs() <= half_w * d) & (dist % d == 0)
                
            masks.append(mask_h.unsqueeze(0)) # Shape (1, seq_len, seq_len)
            
        # Stack to shape (1, h, seq_len, seq_len)
        attn_mask = torch.stack(masks, dim=0).transpose(0, 1) # (1, h, seq_len, seq_len)
        
        # Incorporate global attention if present
        if global_attention_mask is not None:
            # global_attention_mask is shape (B, seq_len) with 1s at global indices
            # Reshape global attention mask for broadcasting
            B = global_attention_mask.size(0)
            g_mask_row = global_attention_mask.view(B, 1, seq_len, 1) # attends to everyone
            g_mask_col = global_attention_mask.view(B, 1, 1, seq_len) # everyone attends to it
            
            # Global attention is symmetric: active if either row or col is global
            is_global = (g_mask_row == 1) | (g_mask_col == 1) # (B, 1, seq_len, seq_len)
            
            # Combine local sliding window mask and global mask
            # If a position is global or satisfies local constraints, it is allowed (True)
            attn_mask = attn_mask | is_global
            
        return attn_mask

    def forward(self, x, global_attention_mask=None):
        """
        Forward pass of Longformer Multi-Head Attention.
        
        Args:
            x (Tensor): Input embedding tensor of shape (B, T, dim)
            global_attention_mask (Tensor, optional): Binary mask of shape (B, T) 
                where 1 indicates global attention at that index.
                
        Returns:
            Tensor: Output embedding of shape (B, T, dim)
        """
        B, T, dim = x.shape
        assert dim == self.dim, f"Input dim {dim} must match expected dim {self.dim}"
        
        # 1. Project Q, K, V for sliding window and global attention
        Qs = self.q_proj_s(x)
        Ks = self.k_proj_s(x)
        Vs = self.v_proj_s(x)
        
        Qg = self.q_proj_g(x)
        Kg = self.k_proj_g(x)
        Vg = self.v_proj_g(x)
        
        # 2. Split into multiple heads: (B, T, h, d_h)
        Qs = Qs.view(B, T, self.num_heads, self.head_dim)
        Ks = Ks.view(B, T, self.num_heads, self.head_dim)
        Vs = Vs.view(B, T, self.num_heads, self.head_dim)
        
        Qg = Qg.view(B, T, self.num_heads, self.head_dim)
        Kg = Kg.view(B, T, self.num_heads, self.head_dim)
        Vg = Vg.view(B, T, self.num_heads, self.head_dim)
        
        # 3. Blending local and global projections
        # If global_attention_mask is provided, blend Q, K, V
        if global_attention_mask is not None:
            # Reshape global attention mask to (B, T, 1, 1) for broadcasting
            g_mask = global_attention_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=x.dtype)
            
            # Symmetrically assign Qg, Kg, Vg to global locations
            Q = g_mask * Qg + (1.0 - g_mask) * Qs
            K = g_mask * Kg + (1.0 - g_mask) * Ks
            V = g_mask * Vg + (1.0 - g_mask) * Vs
        else:
            Q, K, V = Qs, Ks, Vs
            
        # 4. Transpose to (B, h, T, d_h) for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 5. Compute full attention scores: (B, h, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 6. Create and apply the custom Longformer attention mask
        attn_mask = self._create_attention_mask(T, x.device, global_attention_mask)
        # Convert boolean mask to float logit mask (-inf for masked out locations)
        logit_mask = torch.zeros_like(scores)
        logit_mask = logit_mask.masked_fill(~attn_mask, float('-inf'))
        
        # Apply mask
        scores = scores + logit_mask
        
        # 7. Compute Softmax and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Avoid NaN weights for fully masked rows (just in case)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # 8. Compute weighted average of Values: (B, h, T, d_h)
        context = torch.matmul(attn_weights, V)
        
        # 9. Concatenate heads and apply final output projection
        context = context.transpose(1, 2).contiguous().view(B, T, dim)
        output = self.out_proj(context)
        
        return output
