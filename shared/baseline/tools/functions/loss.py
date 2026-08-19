import torch
from torch import Tensor

def cross_entropy_loss(predicted_logits: Tensor, targets: Tensor):
    """
    predicted_logits: [B, S, V] ou [N, V]
    targets:          [B, S]     ou [N]   (dtype long, valores em [0, V-1])
    """
    # Se vier em 3D [B, S, V], achata para [N, V]; targets vira [N]
    if predicted_logits.dim() == 3:
        B, S, V = predicted_logits.shape
        logits = predicted_logits.reshape(B * S, V)
        t = targets.reshape(B * S)
    else:
        logits = predicted_logits
        t = targets

    # log_softmax faz o LogSumExp trick internamente de forma estável
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    t = t.long()
    
    # Negative Log Likelihood
    selected = log_probs.gather(dim=-1, index=t.unsqueeze(-1)).squeeze(-1)
    loss = -torch.mean(selected)

    return loss