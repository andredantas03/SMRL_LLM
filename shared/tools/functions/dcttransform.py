import torch
import math

class DCTTransform:
    """
    Constructs and applies the Discrete Cosine Transform (DCT-II with orthonormal scaling)
    along mode-3 (tube dimension) of a third-order tensor.
    """
    @staticmethod
    def get_matrix(p, device=None, dtype=torch.float32, kind="dct"):
        """
        Computes the orthonormal DCT-II matrix of shape (p, p).
        """
        if kind == "identity":
            return DCTTransform.identity(p)
        elif kind == "dct":
            return DCTTransform.dct(p, device=device, dtype=dtype)
        elif kind == "dst":
            return DCTTransform.dst(p, device=device, dtype=dtype)
        elif kind == "dht":
            return DCTTransform.dht(p, device=device, dtype=dtype)
        elif kind == "haar":
            return DCTTransform.haar(p, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid kind: {kind}")

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
        return torch.einsum('b t s q, q p -> b t s p', A_hat, Z.T)

    @staticmethod
    def identity(p, device=None, dtype=torch.float32):
        return torch.eye(p, device=device, dtype=dtype)
    
    @staticmethod
    def dct(p, device=None, dtype=torch.float32):
        Z = torch.zeros(p, p, device=device, dtype=dtype)
        for u in range(p):
            for v in range(p):
                angle = (math.pi * (2 * v + 1) * u) / (2 * p)
                scale = math.sqrt(1.0 / p) if u == 0 else math.sqrt(2.0 / p)
                Z[u, v] = scale * math.cos(angle)
        return Z

    @staticmethod
    def dst(p, device=None, dtype=torch.float32):
        S = torch.zeros((p, p), device=device, dtype=dtype)
        for n in range(p):
            for k in range(p):
                S[n, k] = math.sin((2 * n + 1) * (k + 1) * math.pi / (2 * p))
        
        # Orthonormal scaling factors
        b = torch.ones(p, device=device, dtype=dtype) * math.sqrt(2.0 / p)
        b[p - 1] = math.sqrt(1.0 / p)
        
        # Scale each column k by b[k]
        S = S * b.unsqueeze(0)
        return S

    @staticmethod
    def dht(p, device=None, dtype=torch.float32):
        """
        Constructs an orthonormal Discrete Hartley Transform (DHT) matrix.
        Formulated as the real part minus the imaginary part of the normalized DFT matrix.
        H = Re(W) - Im(W). Symmetric and orthogonal: H = H.T = H^-1.
        """
        # Create the normalized DFT matrix
        grid_n = torch.arange(p, device=device, dtype=dtype).unsqueeze(1)
        grid_k = torch.arange(p, device=device, dtype=dtype).unsqueeze(0)
        angles = -2 * math.pi * torch.matmul(grid_n, grid_k) / p
        
        # H = Re(W) - Im(W) = (cos(angle) - sin(angle)) / sqrt(p)
        H = (torch.cos(angles) - torch.sin(angles)) / math.sqrt(p)
        return H

    @staticmethod
    def haar(p, device=None, dtype=torch.float32):
        """
        Constructs an orthonormal Discrete Haar Transform matrix of size p (p must be a power of 2).
        Satisfies H_T @ H_T.T = I.
        """
        assert (p & (p - 1)) == 0 and p > 0, "p must be a power of 2 for Haar Transform"
        H_T = torch.zeros((p, p), device=device, dtype=dtype)
        H_T[0, :] = 1.0 / math.sqrt(p)
        for k in range(1, p):
            p_log2 = int(math.floor(math.log2(k)))
            q = k - 2**p_log2
            L = p // (2**p_log2)
            start = q * L
            mid = start + L // 2
            end = start + L
            val = math.sqrt(2**p_log2) / math.sqrt(p)
            H_T[k, start:mid] = val
            H_T[k, mid:end] = -val
        return H_T