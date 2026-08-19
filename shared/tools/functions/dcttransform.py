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
        Z = torch.zeros(p, p, device=device, dtype=dtype)
        if kind == "identity":
            return torch.eye(p, device=device, dtype=dtype)
        elif kind == "dct":
            for u in range(p):
                for v in range(p):
                    angle = (math.pi * (2 * v + 1) * u) / (2 * p)
                    scale = math.sqrt(1.0 / p) if u == 0 else math.sqrt(2.0 / p)
                    Z[u, v] = scale * math.cos(angle)
            return Z
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
        return torch.einsum('b t s q, q p -> b t s p', A_hat, Z)