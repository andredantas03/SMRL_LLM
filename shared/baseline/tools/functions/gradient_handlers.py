from collections.abc import Callable, Iterable, Iterator
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):

    # Reune todos os parametros em uma lista, excluindo parametros vazios e sem gradiente
    # Isso é feito para se calcular o gradiente Global
    params = [p for p in parameters if p is not None and p.grad is not None]

    if not params:
        return torch.tensor(0.0)

    total_sq = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)

    for p in params:
        g = p.grad
        # Se g é esparça, entao o calculo deve ser feito valor a valor(operação feita para ganho de performance)
        if g.is_sparse:
            v = g.coalesce()._values().float()
            total_sq += (v * v).sum()

        else:
            gf = g.float()
            total_sq += (gf * gf).sum()

    # ℓ2-norm ∥g∥2 de todos os parametros:
    total_norm = total_sq.sqrt()

    # Realiza a comparação, se max_norm >= ℓ2-norm ∥g∥2, então o clip_coef recebe 1
    clip_coef = (max_l2_norm / (total_norm + 1e-6)).clamp(max=1.0)

    # tensor.item() Retorna o valor deste tensor como um número padrão do Python. Isso só funciona para tensores com um elemento.
    if clip_coef.item() < 1.0:
        with torch.no_grad():
            for p in params:
                g = p.grad
                if g.is_sparse:
                    g._values().mul_(clip_coef)
                else:
                    g.mul_(clip_coef)
    return None