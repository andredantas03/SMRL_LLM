import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_features=256,
        ortho_features=True,
        causal=False,
        feature_type="hyperbolic",
        numerical_stabilizer=1e-6,
        dropout=0.1
    ):
        """
        Performer Multi-Head Self-Attention using FAVOR+ (Fast Attention Via positive Orthogonal Random features).
        
        Args:
            dim (int): Input embedding dimension (d_m)
            num_heads (int): Number of attention heads (h)
            num_features (int): Number of random features (m). If feature_type is "hyperbolic",
                                the mapped feature dimension will be 2 * num_features.
            ortho_features (bool): If True, use Orthogonal Random Features (ORFs). If False, use IID Gaussian.
            causal (bool): If True, run in unidirectional (causal/autoregressive) mode.
            feature_type (str): Type of positive random feature mapping:
                - "standard": Standard exponential positive random features (l=1, r=m)
                - "hyperbolic": Hyperbolic cosine positive random features (l=2, r=2m)
            numerical_stabilizer (float): Tiny value added to denominator to prevent division by zero or negative values.
            dropout (float): Dropout rate applied to the projections.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_features = num_features
        self.ortho_features = ortho_features
        self.causal = causal
        self.feature_type = feature_type.lower()
        self.numerical_stabilizer = numerical_stabilizer
        
        # Projections to Q, K, V subspaces
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # We will register the random projection weights as a buffer to avoid treating them as parameters,
        # but allowing them to be moved to GPU automatically when .to(device) is called.
        # Shape of random projection matrix for each head: (num_heads, num_features, head_dim)
        self.register_buffer("random_projections", self._generate_random_projections())
        
    def _generate_random_projections(self):
        """
        Generates random projection vectors for each head.
        If ortho_features=True, it generates Orthogonal Random Features (ORFs).
        Otherwise, it generates standard IID Gaussian random features.
        """
        # Shape: (h, m, d_h)
        projections = torch.empty(self.num_heads, self.num_features, self.head_dim)
        
        for h in range(self.num_heads):
            if self.ortho_features:
                # Orthogonal Random Features (ORFs)
                # We generate orthogonal blocks of size (head_dim, head_dim)
                blocks = []
                num_blocks = math.ceil(self.num_features / self.head_dim)
                for _ in range(num_blocks):
                    # Step 1: Generate a random Gaussian matrix
                    G = torch.randn(self.head_dim, self.head_dim)
                    # Step 2: Perform QR decomposition to get orthogonal Q
                    Q, R = torch.linalg.qr(G)
                    # Step 3: We want to preserve the marginal Gaussian norm.
                    # Each row of standard normal matrix has chi-distributed norm.
                    # S is drawn as the row-wise norms of a standard normal matrix.
                    S = torch.randn(self.head_dim, self.head_dim).norm(dim=1)
                    # Step 4: Scale the rows of Q by the norms S
                    block = torch.diag(S) @ Q
                    blocks.append(block)
                # Concatenate blocks and slice to get exactly num_features rows
                W = torch.cat(blocks, dim=0)[:self.num_features]
                projections[h] = W
            else:
                # Standard IID Gaussian features
                projections[h] = torch.randn(self.num_features, self.head_dim)
                
        return projections

    def redraw_features(self, device=None):
        """
        Cheaps redrawing of the random features to prevent getting stuck in poor projections,
        as recommended in the Performer paper (Section 4.2).
        """
        device = device if device is not None else self.random_projections.device
        new_projections = self._generate_random_projections().to(device)
        self.random_projections.copy_(new_projections)

    def _get_positive_features(self, X):
        """
        Transforms input query or key tensor into positive random feature space using FAVOR+.
        Input X shape: (B, h, T, d_h)
        Output shape: (B, h, T, r) where r is the feature dimension (m or 2m).
        """
        B, h, T, d_h = X.shape
        m = self.num_features
        
        # 1. Scale input to handle the sqrt(d_h) term in SM(x,y) = exp(q^T k / sqrt(d_h))
        X_scaled = X / (d_h ** 0.25)
        
        # 2. Project onto random features: (B, h, T, d_h) x (h, d_h, m) -> (B, h, T, m)
        # We need to project each head's data with its corresponding random projection matrix.
        # self.random_projections has shape (h, m, d_h). Let's transpose it to (h, d_h, m)
        W = self.random_projections.transpose(1, 2) # (h, d_h, m)
        
        # Projection using einsum:
        # B = batch, h = heads, t = tokens, d = head_dim, m = features
        projected = torch.einsum('b h t d, h d m -> b h t m', X_scaled, W)
        
        # 3. Compute norm squared along head_dim to apply the exponential scale factor h(x)
        # norm_squared shape: (B, h, T, 1)
        norm_squared = (X_scaled ** 2).sum(dim=-1, keepdim=True)
        h_x = torch.exp(-norm_squared / 2.0)
        
        # 4. Map projected features to positive space using exponential functions (PRFs)
        if self.feature_type == "standard":
            # SM+_m: h(x)/sqrt(m) * exp(u)
            # Size r = m
            mapped = h_x * torch.exp(projected) / math.sqrt(m)
        elif self.feature_type == "hyperbolic":
            # SM_hyp+_m: 1/sqrt(2m) * h(x) * [exp(u), exp(-u)]
            # Size r = 2m
            exp_pos = torch.exp(projected)
            exp_neg = torch.exp(-projected)
            # Concatenate along the feature dimension to get 2m features
            concat_features = torch.cat([exp_pos, exp_neg], dim=-1)
            mapped = h_x * concat_features / math.sqrt(2.0 * m)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
            
        return mapped

    def forward(self, x):
        """
        Forward pass of FAVOR+ attention.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, dim)
        """
        B, T, dim = x.shape
        h = self.num_heads
        d_h = self.head_dim
        
        # 1. Linear projections
        Q = self.q_proj(x).view(B, T, h, d_h).transpose(1, 2) # (B, h, T, d_h)
        K = self.k_proj(x).view(B, T, h, d_h).transpose(1, 2) # (B, h, T, d_h)
        V = self.v_proj(x).view(B, T, h, d_h).transpose(1, 2) # (B, h, T, d_h)
        
        # 2. Map Queries and Keys to positive random features
        # Q_prime, K_prime shape: (B, h, T, r) where r = m (standard) or 2m (hyperbolic)
        Q_prime = self._get_positive_features(Q)
        K_prime = self._get_positive_features(K)
        
        if not self.causal:
            # --- Bidirectional/Masked Attention (O(T) time and space) ---
            # 3. Compute G = (K')^T V of shape (B, h, r, d_h)
            # einsum notation:
            # B = batch, h = head, t = seq_len, r = random_features, d = head_dim
            G = torch.einsum('b h t r, b h t d -> b h r d', K_prime, V)
            
            # 4. Compute numerator: Q' G of shape (B, h, T, d_h)
            num = torch.einsum('b h t r, b h r d -> b h t d', Q_prime, G)
            
            # 5. Compute denominator: Q' ( (K')^T 1_T )
            # K_prime_sum shape: (B, h, r)
            K_prime_sum = K_prime.sum(dim=2) # sum over sequence dimension T
            den = torch.einsum('b h t r, b h r -> b h t', Q_prime, K_prime_sum).unsqueeze(-1)
            
            # 6. Normalize attention output
            out = num / (den + self.numerical_stabilizer)
            
        else:
            # --- Unidirectional Causal Attention (O(T) time and space using parallel Prefix Sums) ---
            # 3. Outer product K_prime_t * V_t shape: (B, h, T, r, d_h)
            # We construct this efficiently using broadcasting: (B, h, T, r, 1) * (B, h, T, 1, d_h)
            outer = K_prime.unsqueeze(-1) * V.unsqueeze(-2)
            
            # 4. Compute prefix sums (cumulative sum along sequence dimension T)
            # GPS shape: (B, h, T, r, d_h)
            GPS = torch.cumsum(outer, dim=2)
            
            # 5. Compute numerator at each step t: Q'_t * GPS_t
            # einsum notation:
            # B = batch, h = head, t = token, r = random_features, d = head_dim
            num = torch.einsum('b h t r, b h t r d -> b h t d', Q_prime, GPS)
            
            # 6. Compute prefix sum of K_prime for normalizer denominator
            # K_prime_cumsum shape: (B, h, T, r)
            K_prime_cumsum = torch.cumsum(K_prime, dim=2)
            # den shape: (B, h, T, 1)
            den = (Q_prime * K_prime_cumsum).sum(dim=-1, keepdim=True)
            
            # 7. Normalize
            out = num / (den + self.numerical_stabilizer)
            
        # 8. Concatenate heads and apply output projection
        # out shape: (B, h, T, d_h) -> Transpose to (B, T, h, d_h) -> View to (B, T, dim)
        out = out.transpose(1, 2).contiguous().view(B, T, dim)
        out = self.out_proj(out)
        return self.dropout(out)
