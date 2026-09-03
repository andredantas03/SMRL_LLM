from einops import einsum, rearrange
import torch

def l_transform(A, Z):
    Y = einsum(A, Z, '... t d p, q p ->... t d q')
    return Y

def l_transform_inverse(A, Z):
    Y = einsum(A, Z.T, '... t d p, q p ->... t d q')
    return Y

def facewise_product(A, B):
    C = einsum(A, B, '... m l p, l n p -> ... m n p')
    return C

def l_product(A, W, Z):
    A_hat = l_transform(A, Z)
    W_hat = l_transform(W, Z)
    C = facewise_product(A_hat, W_hat)
    C = l_transform_inverse(C, Z)
    return C
    
    
    
        