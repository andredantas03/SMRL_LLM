import math
import torch
import torch.nn as nn

class EmbeddingTensorizer(nn.Module):
    """
    Handles folding and unfolding of token embeddings into third-order tensors.
    
    Transforms standard matrices of shape (B, T, d) into third-order tensors of shape (B, T, d_s, p),
    where p is the number of slices and d_s = d / p.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def tenp(self, X):
        """
        Reshapes a 2D sequence representation (plus batch dim) into a 3D tensor.
        X shape: (B, T, d) -> Output shape: (B, T, d_s, p)
        """
        B, T, d = X.shape
        assert d % self.p == 0, f"Embedding dimension {d} must be divisible by decomposition factor p={self.p}"
        d_s = d // self.p
        # View as (B, T, p, d_s) and transpose modes 2 and 3 to get (B, T, d_s, p)
        return X.view(B, T, self.p, d_s).transpose(2, 3).contiguous()

    def matp(self, X_tensor):
        """
        Reconstructs the original 2D matrix shape from the 3D tensor representation.
        X_tensor shape: (B, T, d_s, p) -> Output shape: (B, T, d)
        """
        B, T, d_s, p = X_tensor.shape
        # Transpose back and view as (B, T, d_s * p)
        return X_tensor.transpose(2, 3).contiguous().view(B, T, p * d_s)


class DCTTransform:
    """
    Constructs and applies the Discrete Cosine Transform (DCT-II with orthonormal scaling)
    along mode-3 (tube dimension) of a third-order tensor.
    """
    @staticmethod
    def get_matrix(p, device=None, dtype=torch.float32):
        """
        Computes the orthonormal DCT-II matrix of shape (p, p).
        """
        Z = torch.zeros(p, p, device=device, dtype=dtype)
        for u in range(p):
            for v in range(p):
                angle = (math.pi * (2 * v + 1) * u) / (2 * p)
                scale = math.sqrt(1.0 / p) if u == 0 else math.sqrt(2.0 / p)
                Z[u, v] = scale * math.cos(angle)
        return Z

    @staticmethod
    def forward(A, Z):
        """
        L(A) = A x_3 Z
        A shape: (B, T, d_s, p)
        Z shape: (p, p)
        """
        return torch.einsum('b t s p, q p -> b t s q', A, Z)

    @staticmethod
    def inverse(A_hat, Z):
        """
        L^-1(A_hat) = A_hat x_3 Z^-1
        Since Z is orthonormal, Z^-1 = Z^T.
        """
        return torch.einsum('b t s q, q p -> b t s p', A_hat, Z)


class SliceAwarePositionalEncoding(nn.Module):
    """
    Implements Slice-Aware Sinusoidal Positional Encoding (Definition 5.3).
    Includes fixed (Linear, Exponential, Harmonic) and Learnable frequency scaling strategies.
    """
    def __init__(self, T_max, d_s, p, strategy="linear"):
        super().__init__()
        self.T_max = T_max
        self.d_s = d_s
        self.p = p
        self.strategy = strategy.lower()

        # Compute fixed alpha_k factors
        if self.strategy == "linear":
            alpha = torch.tensor([(k + 1) / p for k in range(p)])
        elif self.strategy == "exponential":
            if p > 1:
                alpha = torch.tensor([2 ** (k / (p - 1)) for k in range(p)])
            else:
                alpha = torch.tensor([1.0])
        elif self.strategy == "harmonic":
            alpha = torch.tensor([float(k + 1) for k in range(p)])
        elif self.strategy == "learnable":
            # Treat scaling factors as trainable parameters
            self.alpha = nn.Parameter(torch.ones(p))
        elif self.strategy == "standard":
            alpha = torch.ones(p)
        else:
            raise ValueError(f"Unknown frequency scaling strategy: {strategy}")

        if self.strategy != "learnable":
            self.register_buffer("alpha", alpha)

        # Compute base sinusoidal components
        pe = torch.zeros(T_max, d_s, p)
        # 10000^(2 * floor(j/2) / d_s)
        div_term = torch.exp(torch.arange(0, d_s, 2).float() * -(math.log(10000.0) / d_s)) # (d_s // 2)
        
        self.register_buffer("div_term", div_term)
        self.register_buffer("pe_template", pe)

    def forward(self, batch_size, T, device):
        # We compute the slice-aware positional encoding dynamically up to seq length T
        pe = torch.zeros(T, self.d_s, self.p, device=device, dtype=self.alpha.dtype)
        t = torch.arange(T, device=device, dtype=self.alpha.dtype).unsqueeze(1) # (T, 1)

        # Iterate over slices to apply alpha_k frequency scaling
        for k in range(self.p):
            scaled_t = t * self.alpha[k]
            # Even indices use sine, odd indices use cosine
            pe[:, 0::2, k] = torch.sin(scaled_t * self.div_term)
            pe[:, 1::2, k] = torch.cos(scaled_t * self.div_term)

        # Expand to batch dimension: (B, T, d_s, p)
        return pe.unsqueeze(0).expand(batch_size, -1, -1, -1)


class TensorMultiHeadAttention(nn.Module):
    """
    Implements L-Multi-Head Attention (Algorithm 1 and 2).
    Optimized to treat the slice index as a batch dimension (Parallel implementation).
    """
    def __init__(self, d_s, p, h):
        super().__init__()
        self.d_s = d_s
        self.p = p
        self.h = h
        assert d_s % h == 0, f"Slice width d_s={d_s} must be divisible by heads h={h}"
        self.d_h = d_s // h

        # Trainable parameters stored directly in transform domain
        # Shapes: (p, d_s, d_s) for projection matrices within each slice
        self.W_q = nn.Parameter(torch.empty(p, d_s, d_s))
        self.W_k = nn.Parameter(torch.empty(p, d_s, d_s))
        self.W_v = nn.Parameter(torch.empty(p, d_s, d_s))
        self.W_o = nn.Parameter(torch.empty(p, d_s, d_s))

        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for projections
        for W in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    def forward(self, X_pos, Z):
        # 1. Transform input activations to DCT domain: (B, T, d_s, p)
        X_hat = DCTTransform.forward(X_pos, Z)

        # 2. Treat slice index as a batch dimension for GPU concurrency:
        # Permute (B, T, d_s, p) -> (B, p, T, d_s)
        X_hat_sliced = X_hat.permute(0, 3, 1, 2)
        B, p, T, d_s = X_hat_sliced.shape

        # 3. Project queries, keys, and values within each slice
        # X_hat_sliced is (B, p, T, d_s), self.W_* is (p, d_s, d_s)
        Q = torch.einsum('b p t s, p s d -> b p t d', X_hat_sliced, self.W_q)
        K = torch.einsum('b p t s, p s d -> b p t d', X_hat_sliced, self.W_k)
        V = torch.einsum('b p t s, p s d -> b p t d', X_hat_sliced, self.W_v)

        # 4. Split into h attention heads: (B, p, T, h, d_h) -> Transpose to (B, p, h, T, d_h)
        Q = Q.view(B, p, T, self.h, self.d_h).transpose(2, 3)
        K = K.view(B, p, T, self.h, self.d_h).transpose(2, 3)
        V = V.view(B, p, T, self.h, self.d_h).transpose(2, 3)

        # 5. Fuse slice and head dimensions to run standard batched attention
        # Shape: (B, p * h, T, d_h)
        Q_fused = Q.reshape(B, p * self.h, T, self.d_h)
        K_fused = K.reshape(B, p * self.h, T, self.d_h)
        V_fused = V.reshape(B, p * self.h, T, self.d_h)

        # 6. Scaled Dot-Product Attention inside slices
        scores = torch.matmul(Q_fused, K_fused.transpose(-2, -1)) / math.sqrt(self.d_h) # (B, p * h, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V_fused) # (B, p * h, T, d_h)

        # 7. Unfuse dimensions and concatenate heads
        # Shape: (B, p, h, T, d_h) -> (B, p, T, h, d_h) -> (B, p, T, d_s)
        context = context.view(B, p, self.h, T, self.d_h).transpose(2, 3)
        H = context.reshape(B, p, T, d_s)

        # 8. Output projection within slice: (B, p, T, d_s)
        Y_hat_sliced = torch.einsum('b p t s, p s d -> b p t d', H, self.W_o)

        # 9. Form Y_hat by permuting back to frontal slices: (B, T, d_s, p)
        Y_hat = Y_hat_sliced.permute(0, 2, 3, 1).contiguous()

        # 10. Map back using inverse L-transform
        Y = DCTTransform.inverse(Y_hat, Z)
        return Y


class TensorFFN(nn.Module):
    """
    Implements L-Feed-Forward Network (Definition 5.5 / Theorem 5.6).
    Applies standard FFNs independently to each transform-domain slice.
    """
    def __init__(self, d_s, d_ff_s, p, activation="relu"):
        super().__init__()
        self.d_s = d_s
        self.d_ff_s = d_ff_s
        self.p = p

        # Transform-domain weight parameter tensors
        self.W1 = nn.Parameter(torch.empty(p, d_s, d_ff_s))
        self.W2 = nn.Parameter(torch.empty(p, d_ff_s, d_s))
        self.b1 = nn.Parameter(torch.empty(p, 1, d_ff_s))
        self.b2 = nn.Parameter(torch.empty(p, 1, d_s))

        if activation == "relu":
            self.act = torch.relu
        elif activation == "gelu":
            self.act = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.reset_parameters()

    def reset_parameters(self):
        for W in [self.W1, self.W2]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, X, Z):
        # 1. Transform activations to spectral domain: (B, T, d_s, p)
        X_hat = DCTTransform.forward(X, Z)

        # 2. Reshape to treat slice mode as batch: (B, p, T, d_s)
        X_hat_sliced = X_hat.permute(0, 3, 1, 2)

        # 3. Apply first linear layer: H = X_hat * W1 + b1
        # unsqueeze(0) transforms (p, 1, d_ff_s) to (1, p, 1, d_ff_s) to broadcast over Batch size B
        H = torch.einsum('b p t s, p s f -> b p t f', X_hat_sliced, self.W1) + self.b1.unsqueeze(0)

        # 4. Element-wise non-linearity
        G = self.act(H)

        # 5. Apply second linear layer: Y = G * W2 + b2
        # unsqueeze(0) transforms (p, 1, d_s) to (1, p, 1, d_s) to broadcast over Batch size B
        Y_hat_sliced = torch.einsum('b p t f, p f s -> b p t s', G, self.W2) + self.b2.unsqueeze(0)

        # 6. Permute back and transform to original domain
        Y_hat = Y_hat_sliced.permute(0, 2, 3, 1).contiguous()
        return DCTTransform.inverse(Y_hat, Z)


class TensorLayerNorm(nn.Module):
    """
    Implements Tensor Layer Normalization (Definition 5.7).
    Computes mean and variance along feature mode-2 slice-by-slice.
    """
    def __init__(self, d_s, p, eps=1e-5):
        super().__init__()
        self.d_s = d_s
        self.p = p
        self.eps = eps

        # Gamma and Beta parameters are defined slice-by-slice
        self.gamma = nn.Parameter(torch.ones(1, 1, d_s, p))
        self.beta = nn.Parameter(torch.zeros(1, 1, d_s, p))

    def forward(self, X):
        # X shape: (B, T, d_s, p)
        mean = X.mean(dim=2, keepdim=True) # mean along mode-2
        var = X.var(dim=2, keepdim=True, unbiased=False) # variance along mode-2
        
        X_norm = (X - mean) / torch.sqrt(var + self.eps)
        return self.gamma * X_norm + self.beta


class TensorEncoderLayer(nn.Module):
    """
    Implements a single Tensor Transformer Encoder Layer (Definition 5.8).
    Supports Post-LayerNorm (as in paper equations) or standard Pre-LayerNorm.
    """
    def __init__(self, d, p, h, d_ff, activation="relu", eps=1e-5, norm_first=False):
        super().__init__()
        self.d_s = d // p
        self.p = p
        self.norm_first = norm_first

        self.mhal_layer = TensorMultiHeadAttention(self.d_s, p, h)
        self.ffn_layer = TensorFFN(self.d_s, self.d_s * 4, p, activation=activation)
        self.tln1 = TensorLayerNorm(self.d_s, p, eps=eps)
        self.tln2 = TensorLayerNorm(self.d_s, p, eps=eps)

    def forward(self, X, P, Z):
        if self.norm_first:
            # Pre-LN variant
            X_norm = self.tln1(X)
            X_pos = X_norm + P if P is not None else X_norm
            X = X + self.mhal_layer(X_pos, Z)
            
            X_norm2 = self.tln2(X)
            X = X + self.ffn_layer(X_norm2, Z)
            return X
        else:
            # Post-LN variant (Definition 5.8)
            X_pos = X + P if P is not None else X
            # Equation 5.3: X' = TLN( X + MHAL(X_pos) )
            X_prime = self.tln1(X + self.mhal_layer(X_pos, Z))
            
            # Equation 5.4: Y = TLN( X' + TFFN(X') )
            Y = self.tln2(X_prime + self.ffn_layer(X_prime, Z))
            return Y


class TensorTransformerEncoder(nn.Module):
    """
    A full Stack of N Tensor Encoder Layers (Definition 5.10).
    """
    def __init__(self, num_layers, d, p, h, d_ff, vocab_size, T_max, 
                 pe_strategy="linear", activation="relu", norm_first=False):
        super().__init__()
        self.d = d
        self.p = p
        self.h = h
        self.d_s = d // p
        self.vocab_size = vocab_size

        # Word embeddings in standard 2D space
        self.token_embeddings = nn.Embedding(vocab_size, d)
        
        # Folding / unfolding utilities
        self.tensorizer = EmbeddingTensorizer(p)

        # Slice-aware positional encoding
        self.positional_encoding = SliceAwarePositionalEncoding(T_max, self.d_s, p, strategy=pe_strategy)

        # Core encoder layer stack
        self.layers = nn.ModuleList([
            TensorEncoderLayer(d, p, h, d_ff, activation=activation, norm_first=norm_first)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        # Get the fixed orthogonal DCT-II matrix
        Z = DCTTransform.get_matrix(self.p, device=device, dtype=torch.float32)

        # 1. Standard token embeddings: (B, T, d)
        x_emb = self.token_embeddings(input_ids)

        # 2. Reshape/fold embeddings to tensor space: (B, T, d_s, p)
        X = self.tensorizer.tenp(x_emb)

        # 3. Dynamic positional encoding generation
        P = self.positional_encoding(B, T, device)

        # 4. Forward pass through N sequential tensor encoder blocks
        for layer in self.layers:
            X = layer(X, P, Z)

        # 5. Reconstruct standard representations: (B, T, d)
        return self.tensorizer.matp(X)


class TensorTransformerForSequenceClassification(nn.Module):
    """
    A complete sequence classification model using the Tensor Transformer Encoder.
    """
    def __init__(self, num_layers, d, p, h, d_ff, vocab_size, T_max, num_classes,
                 pe_strategy="linear", activation="relu", norm_first=False):
        super().__init__()
        self.encoder = TensorTransformerEncoder(
            num_layers=num_layers, d=d, p=p, h=h, d_ff=d_ff,
            vocab_size=vocab_size, T_max=T_max, pe_strategy=pe_strategy,
            activation=activation, norm_first=norm_first
        )
        # Sequence classification classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d, num_classes)
        )

    def forward(self, input_ids):
        # 1. Run through the tensorized encoder stack: (B, T, d)
        hidden_states = self.encoder(input_ids)

        # 2. Mean pooling over the sequence tokens: (B, d)
        pooled = hidden_states.mean(dim=1)

        # 3. Linear classification projections: (B, num_classes)
        return self.classifier(pooled)


if __name__ == "__main__":
    # Test instantiation and mock train step
    print("Initializing a complete Tensor Transformer Classifier...")
    model = TensorTransformerForSequenceClassification(
        num_layers=4,
        d=128,
        p=4,
        h=4,
        d_ff=512,
        vocab_size=30000,
        T_max=128,
        num_classes=2,
        pe_strategy="linear"
    )

    # Generate synthetic input IDs
    input_ids = torch.randint(0, 30000, (8, 128)) # Batch size 8, Seq len 128
    labels = torch.randint(0, 2, (8,))

    # Check forward pass
    outputs = model(input_ids)
    print(f"Input batch shape: {input_ids.shape}")
    print(f"Output predictions shape: {outputs.shape}")

    # Check backpropagation
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    loss.backward()
    print("Forward and backward passes completed successfully!")

    # Count total encoder parameters
    encoder_params = sum(param.numel() for param in model.encoder.layers.parameters())
    print(f"Total encoder layer parameters: {encoder_params}")
