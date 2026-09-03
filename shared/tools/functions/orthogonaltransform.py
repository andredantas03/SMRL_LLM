import torch
import math

from shared.modules.learnable_z.learnable_z import LearnableZ

class OrthogonalTransform(torch.nn.Module):
    def __init__(self, p , kind="dct", device=None, dtype=torch.float32, ):
        super().__init__()
        self.kind = kind
        self.p = p
        if self.kind == "learnable":
            self.learnable_z = LearnableZ(self.p, device=device, dtype=dtype)
        else:
            Z = self._build_matrix(p, kind, device=device, dtype=dtype)
            self.register_buffer("Z", Z)
    def get_matrix(self, device=None, dtype=torch.float32):
        if self.kind == "learnable":
            return self.learnable_z.z
        return self.Z
    
    @staticmethod
    def _build_matrix(p, kind, device=None, dtype=torch.float32):
        if kind == "identity":
            return OrthogonalTransform.identity(p, device, dtype)
        if kind == "dct":
            return OrthogonalTransform.dct(p, device, dtype)
        if kind == "dst":
            return OrthogonalTransform.dst(p, device, dtype)
        if kind == "dht":
            return OrthogonalTransform.dht(p, device, dtype)
        if kind == "haar":
            return OrthogonalTransform.haar(p, device, dtype)
        raise ValueError(f"Invalid kind: {kind}")

    

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
        return Z.T

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