import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        num_landmarks=64,
        num_heads=8,
        dropout=0.1,
        num_iterations=6,
        conv_kernel_size=33
    ):
        """
        Nyströmformer Multi-Head Self-Attention.
        
        Args:
            dim (int): Input embedding dimension (d_m)
            seq_len (int): Sequence length of the input (n)
            num_landmarks (int): Number of landmark (Nyström) points (m)
            num_heads (int): Number of attention heads (h)
            dropout (float): Dropout rate for attention weights
            num_iterations (int): Number of iterations for Newton-type pseudoinverse approximation
            conv_kernel_size (int): Kernel size for 1D depthwise convolution skip-connection
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.seq_len = seq_len
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_iterations = num_iterations
        
        # Projections to Q, K, V subspaces
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 1D Depthwise Convolution for the skip connection of value V
        # It operates along the sequence dimension, so we treat dim as channels
        self.v_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            groups=dim,
            bias=True
        )

    def _segment_means(self, X, m):
        """
        Extracts landmark points using Segment-means.
        Averages m segments of the input sequence.
        Input X shape: (B, h, n, d_h)
        Output shape: (B, h, m, d_h)
        """
        B, h, n, d_h = X.shape
        remainder = n % m
        if remainder != 0:
            pad_len = m - remainder
            # Pad the sequence dimension (second to last dimension)
            X = F.pad(X, (0, 0, 0, pad_len))
            n = n + pad_len
            
        l = n // m
        # Reshape to group segment elements: (B, h, m, l, d_h)
        X_reshaped = X.view(B, h, m, l, d_h)
        # Average along the segment length dimension (l)
        return X_reshaped.mean(dim=-2)

    def _approximate_pseudoinverse(self, A_S):
        """
        Computes the Moore-Penrose pseudoinverse using the 3rd-order Newton-type
        iterative approximation described in Lemma 1 and Equation 14.
        Input A_S shape: (B, h, m, m)
        Output shape: (B, h, m, m)
        """
        B, h, m, _ = A_S.shape
        I = torch.eye(m, device=A_S.device, dtype=A_S.dtype).expand_as(A_S)
        
        # 1. Compute 1-norm and infinity-norm of A_S for initialization
        # norm_1: max absolute column sum
        norm_1 = torch.max(torch.sum(torch.abs(A_S), dim=-2), dim=-1).values
        # norm_inf: max absolute row sum
        norm_inf = torch.max(torch.sum(torch.abs(A_S), dim=-1), dim=-1).values
        
        # Initialize Z0 (Pan and Schreiber 1991)
        epsilon = 1e-6
        denominator = (norm_1 * norm_inf).clamp(min=epsilon)
        Z = A_S.transpose(-2, -1) / denominator.unsqueeze(-1).unsqueeze(-1)
        
        # 2. Iteratively refine pseudoinverse approximation using 3rd-order convergence formula
        for _ in range(self.num_iterations):
            AZ = torch.matmul(A_S, Z)
            term1 = 7.0 * I - AZ
            AZ_term1 = torch.matmul(AZ, term1)
            term2 = 15.0 * I - AZ_term1
            AZ_term2 = torch.matmul(AZ, term2)
            term3 = 13.0 * I - AZ_term2
            Z = 0.25 * torch.matmul(Z, term3)
            
        return Z

    def forward(self, x):
        """
        Forward pass of Nyströmformer Attention.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, dim) where T = seq_len
        """
        B, T, dim = x.shape
        assert T == self.seq_len, f"Input sequence length ({T}) must match seq_len ({self.seq_len})"
        
        # 1. Project to Q, K, V
        Q = self.q_proj(x) # (B, T, dim)
        K = self.k_proj(x) # (B, T, dim)
        V = self.v_proj(x) # (B, T, dim)
        
        # Compute 1D Depthwise Convolution on V (skip connection)
        # Input shape to Conv1d: (B, dim, T)
        V_conv = self.v_conv(V.transpose(-2, -1)).transpose(-2, -1) # (B, T, dim)
        
        # 2. Reshape Q, K, V to multi-head format: (B, h, T, d_h)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaling factor for attention
        scale = math.sqrt(self.head_dim)
        
        # 3. Compute landmarks using Segment-means (sMEANS)
        Q_tilde = self._segment_means(Q, self.num_landmarks) # (B, h, m, d_h)
        K_tilde = self._segment_means(K, self.num_landmarks) # (B, h, m, d_h)
        
        # 4. Construct the three sub-matrices for Nyström approximation
        # F_tilde = softmax( Q * K_tilde^T / sqrt(d_h) ) -> (B, h, T, m)
        F_tilde = torch.matmul(Q, K_tilde.transpose(-2, -1)) / scale
        F_tilde = F_tilde.softmax(dim=-1)
        F_tilde = self.dropout(F_tilde)
        
        # B_tilde = softmax( Q_tilde * K^T / sqrt(d_h) ) -> (B, h, m, T)
        B_tilde = torch.matmul(Q_tilde, K.transpose(-2, -1)) / scale
        B_tilde = B_tilde.softmax(dim=-1)
        B_tilde = self.dropout(B_tilde)
        
        # A_tilde = softmax( Q_tilde * K_tilde^T / sqrt(d_h) ) -> (B, h, m, m)
        A_tilde = torch.matmul(Q_tilde, K_tilde.transpose(-2, -1)) / scale
        A_tilde = A_tilde.softmax(dim=-1)
        
        # 5. Approximate the Moore-Penrose pseudoinverse A_tilde^+ (pINV)
        A_tilde_pinv = self._approximate_pseudoinverse(A_tilde) # (B, h, m, m)
        
        # 6. Reconstruct the attention output (Eq. 17)
        # ŜV = F_tilde * A_tilde_pinv * B_tilde * V
        # Order of multiplication to maintain O(n) complexity:
        # First compute (B_tilde * V) -> (B, h, m, d_h)
        # Then compute (A_tilde_pinv * (B_tilde * V)) -> (B, h, m, d_h)
        # Finally compute (F_tilde * (A_tilde_pinv * B_tilde * V)) -> (B, h, T, d_h)
        B_V = torch.matmul(B_tilde, V) # (B, h, m, d_h)
        pinv_B_V = torch.matmul(A_tilde_pinv, B_V) # (B, h, m, d_h)
        context = torch.matmul(F_tilde, pinv_B_V) # (B, h, T, d_h)
        
        # 7. Concatenate heads back to original shape: (B, T, dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.dim)
        
        # 8. Add the depthwise conv skip-connection of V
        output = context + V_conv # (B, T, dim)
        
        # 9. Apply output projection
        output = self.out_proj(output)
        
        return output
