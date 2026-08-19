import torch.nn as nn

class SMRL_LM_Head(nn.Module):
    """
    A Language Model Head for the SMRL Transformer Encoder.
    """
    def __init__(self, d, p, vocab_size):
        super().__init__()
        self.d = d
        self.p = p
        self.vocab_size = vocab_size

        # Language Model Head
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)
