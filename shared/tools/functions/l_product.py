import torch

def facewise_product(A,B):
    assert A.shape[1] == B.shape[0]
    assert A.shape[2] == B.shape[2]
    return torch.einsum("mlp,lnp->mnp",A,B)

def l_product(A,B,Z):
    assert A.shape[1] == B.shape[0]
    assert A.shape[2] == B.shape[2]
    assert Z.shape[0] == A.shape[1]
    assert Z.shape[1] == B.shape[1]
    A_hat = torch.einsum("mlp,pq->mnq",A,Z)
    B_hat = torch.einsum("mlp,pq->mnq",B,Z)
    C_hat = facewise_product(A_hat,B_hat)
    C = torch.einsum("mnp,pq->mnq",C_hat,Z.T)
    return C