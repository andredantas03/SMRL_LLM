import torch
from einops import einsum, reduce

def compute_dot_product_affinity(
    q: torch.Tensor,
    k: torch.Tensor,
    scaling: float | None = None,
) -> torch.Tensor:

    if scaling is None:
        scaling = q.shape[-1] ** -0.5

    return einsum(q, k, "bs h sq d, bs h sk d -> bs h sq sk") * scaling




