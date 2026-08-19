from torch import Tensor
from torch import exp as torch_exp

def softmax(x: Tensor, dim: int):
    x_max_values, _ = x.max(dim=dim)
    x -= x_max_values.unsqueeze(dim=dim)
    x = torch_exp(x)
    sum_exp = x.sum(dim=dim).unsqueeze(dim=dim)
    return x / sum_exp
