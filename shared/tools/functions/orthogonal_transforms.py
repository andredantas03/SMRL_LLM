import torch
import torch.nn as nn
import math

__all__ = [
    "get_dst_matrix",
    "get_dht_matrix",
    "get_wht_matrix",
    "get_slant_matrix",
    "get_haar_matrix",
    "dst",
    "idst",
    "dht",
    "idht",
    "wht",
    "iwht",
    "slant",
    "islant",
    "haar",
    "ihaar",
    "KarhunenLoeveTransform"
]

# =====================================================================
# 1. Discrete Sine Transform (DST-II) - Orthonormal
# =====================================================================

def get_dst_matrix(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Constructs an orthonormal Discrete Sine Transform (DST-II) matrix.
    Satisfies S.T @ S = S @ S.T = I.
    """
    S = torch.zeros((N, N), device=device, dtype=dtype)
    for n in range(N):
        for k in range(N):
            S[n, k] = math.sin((2 * n + 1) * (k + 1) * math.pi / (2 * N))
    
    # Orthonormal scaling factors
    b = torch.ones(N, device=device, dtype=dtype) * math.sqrt(2.0 / N)
    b[N - 1] = math.sqrt(1.0 / N)
    
    # Scale each column k by b[k]
    S = S * b.unsqueeze(0)
    return S

def dst(X: torch.Tensor) -> torch.Tensor:
    """Performs Discrete Sine Transform along the last dimension."""
    N = X.shape[-1]
    S = get_dst_matrix(N, device=X.device, dtype=X.dtype)
    return torch.matmul(X, S)

def idst(X_spec: torch.Tensor) -> torch.Tensor:
    """Performs Inverse Discrete Sine Transform along the last dimension."""
    N = X_spec.shape[-1]
    S = get_dst_matrix(N, device=X_spec.device, dtype=X_spec.dtype)
    # Since S is orthonormal, S^-1 = S.T
    return torch.matmul(X_spec, S.t())


# =====================================================================
# 2. Discrete Hartley Transform (DHT) - Orthonormal
# =====================================================================

def get_dht_matrix(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Constructs an orthonormal Discrete Hartley Transform (DHT) matrix.
    Formulated as the real part minus the imaginary part of the normalized DFT matrix.
    H = Re(W) - Im(W). Symmetric and orthogonal: H = H.T = H^-1.
    """
    # Create the normalized DFT matrix
    grid_n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    grid_k = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    angles = -2 * math.pi * torch.matmul(grid_n, grid_k) / N
    
    # H = Re(W) - Im(W) = (cos(angle) - sin(angle)) / sqrt(N)
    H = (torch.cos(angles) - torch.sin(angles)) / math.sqrt(N)
    return H

def dht(X: torch.Tensor) -> torch.Tensor:
    """Performs Discrete Hartley Transform along the last dimension."""
    N = X.shape[-1]
    H = get_dht_matrix(N, device=X.device, dtype=X.dtype)
    return torch.matmul(X, H)

def idht(X_spec: torch.Tensor) -> torch.Tensor:
    """Performs Inverse Discrete Hartley Transform along the last dimension (identical to forward DHT)."""
    return dht(X_spec)


# =====================================================================
# 3. Walsh-Hadamard Transform (WHT) - Orthonormal
# =====================================================================

def get_wht_matrix(N: int, ordering: str = "sequency", device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Constructs an orthonormal Walsh-Hadamard Transform matrix of size N (N must be a power of 2).
    Args:
        N: Matrix dimension.
        ordering: "hadamard" (recursive natural order) or "sequency" (sorted by frequency/zero-crossings).
    """
    assert (N & (N - 1)) == 0 and N > 0, "N must be a power of 2 for Walsh-Hadamard Transform"
    
    # Base Kronecker seed
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype) / math.sqrt(2.0)
    current_N = 2
    while current_N < N:
        seed = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype) / math.sqrt(2.0)
        H = torch.kron(seed, H)
        current_N *= 2
        
    if ordering.lower() == "sequency":
        # Calculate sign changes (zero crossings) for each row to reorder
        sign_changes = []
        for i in range(N):
            row = H[i]
            changes = 0
            for j in range(N - 1):
                if row[j] * row[j + 1] < 0:
                    changes += 1
            sign_changes.append((changes, i))
        sign_changes.sort()
        sorted_indices = [idx for _, idx in sign_changes]
        H = H[sorted_indices]
        
    return H

def wht(X: torch.Tensor, ordering: str = "sequency") -> torch.Tensor:
    """Performs Walsh-Hadamard Transform along the last dimension."""
    N = X.shape[-1]
    H = get_wht_matrix(N, ordering=ordering, device=X.device, dtype=X.dtype)
    return torch.matmul(X, H.t())

def iwht(X_spec: torch.Tensor, ordering: str = "sequency") -> torch.Tensor:
    """Performs Inverse Walsh-Hadamard Transform along the last dimension (identical to forward since H is symmetric and orthogonal)."""
    return wht(X_spec, ordering=ordering)


# =====================================================================
# 4. Slant Transform (ST) - Orthonormal
# =====================================================================

def _slant_transform_recursive_vector(x: torch.Tensor) -> torch.Tensor:
    """Internal helper to construct a Slant vector recursively."""
    N = len(x)
    if N == 2:
        u, v = x[0], x[1]
        out = torch.zeros_like(x)
        out[0] = (u + v) / math.sqrt(2)
        out[1] = (u - v) / math.sqrt(2)
        return out
    else:
        y1 = x[:N//2] + x[N//2:]
        y2 = x[:N//2] - x[N//2:]
        y1_trans = _slant_transform_recursive_vector(y1)
        y2_trans = _slant_transform_recursive_vector(y2)
        
        out = torch.zeros_like(x)
        out[:N//2] = y1_trans / math.sqrt(2)
        out[N//2:] = y2_trans / math.sqrt(2)
        
        w = 4 * N * N - 4
        c = math.sqrt(3 * N * N / w)
        s = math.sqrt((N * N - 4) / w)
        
        u = out[N//4].clone()
        v = out[N//2].clone()
        out[N//4] = c * u - s * v
        out[N//2] = s * u + c * v
        return out

def get_slant_matrix(N: int, ordering: str = "sequency", device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Constructs an orthonormal Slant Transform matrix of size N (N must be a power of 2).
    Args:
        N: Matrix dimension.
        ordering: "hadamard" (recursive natural order) or "sequency" (sorted by frequency/zero-crossings).
    """
    assert (N & (N - 1)) == 0 and N > 0, "N must be a power of 2 for Slant Transform"
    I = torch.eye(N, device=device, dtype=dtype)
    S_T = torch.zeros((N, N), device=device, dtype=dtype)
    for i in range(N):
        S_T[:, i] = _slant_transform_recursive_vector(I[:, i])
        
    if ordering.lower() == "sequency":
        # Sort rows of S_T by their zero crossings to achieve sequency ordering as in the textbook
        sign_changes = []
        for i in range(N):
            row = S_T[i]
            changes = 0
            for j in range(N - 1):
                if row[j] * row[j + 1] < -1e-9:
                    changes += 1
                elif abs(row[j]) < 1e-9 and j > 0 and row[j - 1] * row[j + 1] < -1e-9:
                    changes += 1
            sign_changes.append((changes, i))
        sign_changes.sort()
        sorted_indices = [idx for _, idx in sign_changes]
        S_T = S_T[sorted_indices]
        
    return S_T

def slant(X: torch.Tensor, ordering: str = "sequency") -> torch.Tensor:
    """Performs Slant Transform along the last dimension."""
    N = X.shape[-1]
    S = get_slant_matrix(N, ordering=ordering, device=X.device, dtype=X.dtype)
    return torch.matmul(X, S.t())

def islant(X_spec: torch.Tensor, ordering: str = "sequency") -> torch.Tensor:
    """Performs Inverse Slant Transform along the last dimension."""
    N = X_spec.shape[-1]
    S = get_slant_matrix(N, ordering=ordering, device=X_spec.device, dtype=X_spec.dtype)
    # Since S is orthonormal, S^-1 = S.T
    return torch.matmul(X_spec, S)


# =====================================================================
# 5. Discrete Haar Transform (DHT) - Orthonormal
# =====================================================================

def get_haar_matrix(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Constructs an orthonormal Discrete Haar Transform matrix of size N (N must be a power of 2).
    Satisfies H_T @ H_T.T = I.
    """
    assert (N & (N - 1)) == 0 and N > 0, "N must be a power of 2 for Haar Transform"
    H_T = torch.zeros((N, N), device=device, dtype=dtype)
    H_T[0, :] = 1.0 / math.sqrt(N)
    for k in range(1, N):
        p = int(math.floor(math.log2(k)))
        q = k - 2**p
        L = N // (2**p)
        start = q * L
        mid = start + L // 2
        end = start + L
        val = math.sqrt(2**p) / math.sqrt(N)
        H_T[k, start:mid] = val
        H_T[k, mid:end] = -val
    return H_T

def haar(X: torch.Tensor) -> torch.Tensor:
    """Performs Discrete Haar Transform along the last dimension."""
    N = X.shape[-1]
    H = get_haar_matrix(N, device=X.device, dtype=X.dtype)
    return torch.matmul(X, H.t())

def ihaar(X_spec: torch.Tensor) -> torch.Tensor:
    """Performs Inverse Discrete Haar Transform along the last dimension."""
    N = X_spec.shape[-1]
    H = get_haar_matrix(N, device=X_spec.device, dtype=X_spec.dtype)
    # Since H is orthonormal, H^-1 = H.T
    return torch.matmul(X_spec, H)


