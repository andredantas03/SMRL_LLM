from torch import nn
import torch
from shared.data.dataset_loader import load_config


class LearnableZ(nn.Module):
    def __init__(self, p, device=None, dtype=torch.float32):
        super().__init__()
        self.z = nn.Parameter(torch.zeros((p,p),device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.z)