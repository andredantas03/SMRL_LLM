import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        k,
        num_heads=8,
        dropout=0.1,
        sharing_strategy="none",
        projection_type="linear",
        E_proj=None,
        F_proj=None
    ):
        """
        Linformer Multi-Head Self-Attention.
        
        Args:
            dim (int): Input embedding dimension (d_m)
            seq_len (int): Sequence length of the input (n)
            k (int): Projected dimension (k << n)
            num_heads (int): Number of attention heads (h)
            dropout (float): Attention and projection dropout rate
            sharing_strategy (str): Projection sharing strategy:
                - "none": Head i has its own E_i and F_i
                - "headwise": Share E and F across all heads in this layer
                - "key_value": Share E = F across all heads in this layer
                - "layerwise": Projections are passed externally (E_proj and F_proj)
            projection_type (str): Type of projection method:
                - "linear": nn.Linear(seq_len, k) applied to sequence dimension
                - "pooling": AvgPooling/MaxPooling to reduce n to k (parameter-free)
                - "conv": nn.Conv1d to project n to k
            E_proj (nn.Module, optional): Pre-existing projection module for E (for layerwise sharing)
            F_proj (nn.Module, optional): Pre-existing projection module for F (for layerwise sharing)
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.seq_len = seq_len
        self.k = k
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sharing_strategy = sharing_strategy.lower()
        self.projection_type = projection_type.lower()
        
        # Projections to Q, K, V subspaces
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Setup E and F projection matrices based on sharing strategy and projection type
        self._setup_projections(E_proj, F_proj)
        
    def _setup_projections(self, E_proj, F_proj):
        """Helper to initialize E and F projection operators."""
        if self.sharing_strategy == "layerwise":
            assert E_proj is not None, "E_proj must be provided for layerwise sharing"
            self.E = E_proj
            self.F = F_proj if F_proj is not None else E_proj
            return
            
        if self.projection_type == "linear":
            if self.sharing_strategy == "none":
                # Each head has its own E and F of shape [k, n]
                self.E = nn.Parameter(torch.randn(self.num_heads, self.k, self.seq_len))
                self.F = nn.Parameter(torch.randn(self.num_heads, self.k, self.seq_len))
                # Initialization
                nn.init.normal_(self.E, std=1.0 / math.sqrt(self.k))
                nn.init.normal_(self.F, std=1.0 / math.sqrt(self.k))
            elif self.sharing_strategy == "headwise":
                # Single E and F of shape [k, n] shared across all heads in this layer
                self.E = nn.Linear(self.seq_len, self.k, bias=False)
                self.F = nn.Linear(self.seq_len, self.k, bias=False)
            elif self.sharing_strategy == "key_value":
                # Single E = F shared across all heads in this layer
                self.E = nn.Linear(self.seq_len, self.k, bias=False)
                self.F = self.E
            else:
                raise ValueError(f"Unknown sharing strategy: {self.sharing_strategy}")
                
        elif self.projection_type == "pooling":
            # Parameter-free average pooling projection
            assert self.seq_len % self.k == 0, f"seq_len ({self.seq_len}) must be divisible by k ({self.k}) for pooling projection"
            kernel_size = self.seq_len // self.k
            self.E = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
            self.F = self.E
            
        elif self.projection_type == "conv":
            # Convolutional projection
            assert self.seq_len % self.k == 0, f"seq_len ({self.seq_len}) must be divisible by k ({self.k}) for conv projection"
            kernel_size = self.seq_len // self.k
            
            if self.sharing_strategy == "none":
                # Each head has its own separate Conv1D with groups=num_heads
                self.E = nn.Conv1d(self.dim, self.dim, kernel_size=kernel_size, stride=kernel_size, groups=self.num_heads, bias=False)
                self.F = nn.Conv1d(self.dim, self.dim, kernel_size=kernel_size, stride=kernel_size, groups=self.num_heads, bias=False)
            elif self.sharing_strategy == "headwise":
                # Single Conv1D shared across all heads in this layer
                self.E = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
                self.F = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
            elif self.sharing_strategy == "key_value":
                # Single Conv1D shared across all heads and between K and V in this layer
                self.E = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
                self.F = self.E
            else:
                raise ValueError(f"Unknown sharing strategy: {self.sharing_strategy}")
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

    def _apply_linear_projection(self, X, proj_weight):
        """
        Applies linear projection along the sequence dimension.
        Input X shape: (B, h, n, d_h)
        proj_weight: E or F Parameter of shape (h, k, n) or nn.Linear
        """
        if isinstance(proj_weight, nn.Linear):
            # Shape of X: (B, h, n, d_h) -> Transpose to apply linear to 'n'
            # (B, h, d_h, n) -> nn.Linear maps n to k -> (B, h, d_h, k) -> Transpose back to (B, h, k, d_h)
            X_t = X.transpose(-2, -1)
            out = proj_weight(X_t).transpose(-2, -1)
            return out
        elif isinstance(proj_weight, nn.Parameter):
            # Case sharing_strategy = "none" with Parameter of shape (h, k, n)
            # We want to multiply E_i (k, n) with X_i (B, n, d_h) for each head i
            # X shape: (B, h, n, d_h)
            # proj_weight shape: (h, k, n)
            # Let's use torch.einsum: 'b h n d, h k n -> b h k d'
            return torch.einsum('b h n d, h k n -> b h k d', X, proj_weight)
        elif self.sharing_strategy == "layerwise":
            # Layerwise sharing might be passing a custom nn.Linear or parameter
            if isinstance(proj_weight, nn.Linear):
                X_t = X.transpose(-2, -1)
                return proj_weight(X_t).transpose(-2, -1)
            else:
                return torch.einsum('b h n d, h k n -> b h k d', X, proj_weight)
        else:
            raise TypeError("Unsupported projection type for linear mapping")

    def _apply_pooling_projection(self, X, pooling_layer):
        """
        Applies parameter-free pooling projection.
        Input X shape: (B, h, n, d_h)
        """
        B, h, n, d_h = X.shape
        # Average pooling along sequence dimension 'n'
        # Reshape to (B * h, d_h, n) for nn.AvgPool1d
        X_flat = X.permute(0, 1, 3, 2).reshape(B * h, d_h, n)
        out_flat = pooling_layer(X_flat) # (B * h, d_h, k)
        out = out_flat.view(B, h, d_h, self.k).transpose(-2, -1) # (B, h, k, d_h)
        return out

    def _apply_conv_projection(self, X, conv_layer):
        """
        Applies 1D Convolutional projection.
        Input X shape: (B, h, n, d_h)
        conv_layer: nn.Conv1d
        """
        B, h, n, d_h = X.shape
        
        if self.sharing_strategy == "none":
            # Input X is (B, h, n, d_h) -> Permute to (B, h, d_h, n) -> Reshape to (B, dim, n)
            X_full = X.transpose(2, 3).reshape(B, self.dim, n)
            # Apply group convolution (groups=num_heads)
            out_full = conv_layer(X_full) # (B, dim, k)
            # Reshape back to (B, h, d_h, k) -> Permute to (B, h, k, d_h)
            out = out_full.view(B, h, d_h, self.k).transpose(2, 3)
            return out
        else:
            # "headwise" or "key_value"
            # Reshape to (B * h, d_h, n)
            X_flat = X.transpose(2, 3).reshape(B * h, d_h, n)
            out_flat = conv_layer(X_flat) # (B * h, d_h, k)
            # Reshape back to (B, h, d_h, k) -> Permute to (B, h, k, d_h)
            out = out_flat.view(B, h, d_h, self.k).transpose(2, 3)
            return out

    def forward(self, x, E_proj=None, F_proj=None):
        """
        Forward pass of Linformer Attention.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, dim) where T = seq_len
            E_proj, F_proj (nn.Module, optional): Dynamic projection matrices
        """
        B, T, dim = x.shape
        assert T == self.seq_len, f"Input sequence length ({T}) must match seq_len ({self.seq_len})"
        
        # 1. Project Q, K, V
        # Q, K, V shape: (B, T, dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Reshape to multi-head format: (B, h, T, d_h)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply projection E to Key and F to Value to reduce seq_len 'T' (n) to 'k'
        if self.projection_type == "linear":
            # Get projection operators
            E_op = E_proj if E_proj is not None else self.E
            F_op = F_proj if F_proj is not None else self.F
            K_proj = self._apply_linear_projection(K, E_op) # (B, h, k, d_h)
            V_proj = self._apply_linear_projection(V, F_op) # (B, h, k, d_h)
            
        elif self.projection_type == "pooling":
            K_proj = self._apply_pooling_projection(K, self.E)
            V_proj = self._apply_pooling_projection(V, self.F)
            
        elif self.projection_type == "conv":
            K_proj = self._apply_conv_projection(K, self.E)
            V_proj = self._apply_conv_projection(V, self.F)
        
        # 4. Scaled Dot-Product Attention
        # Q shape: (B, h, T, d_h)
        # K_proj shape: (B, h, k, d_h) -> Transpose to (B, h, d_h, k)
        # Q K^T shape: (B, h, T, k)
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Softmax over key/projection dimension 'k'
        attn_weights = F.softmax(scores, dim=-1) # (B, h, T, k)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Context shape: (B, h, T, k) x (B, h, k, d_h) -> (B, h, T, d_h)
        context = torch.matmul(attn_weights, V_proj)
        
        # 5. Concatenate heads and project out
        # Transpose to (B, T, h, d_h) -> Reshape to (B, T, dim)
        context = context.transpose(1, 2).contiguous().view(B, T, dim)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        return output
