import torch.nn as nn
from einops import rearrange,einsum



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff,bias=True)
        self.linear2 = nn.Linear(d_ff, d_model,bias=True)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))





