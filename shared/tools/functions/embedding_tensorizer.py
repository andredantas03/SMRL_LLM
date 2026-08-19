from torch import nn
from einops import rearrange

class EmbeddingTensorizer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def tenp(self, X):        
        bs, s, d = X.shape
        assert d % self.p == 0, f"Embedding dimension {d} must be divisible by decomposition factor p={self.p}"
        ds = d // self.p
        X = rearrange(X, "b s (p ds) -> b s ds p", p=self.p, ds=ds)
        return X

    def matp(self, X_tensor):
        X_mat = rearrange(X_tensor, "b s ds p -> b s (p ds)")
        return X_mat