import os
import typing

import torch
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    cfg: dict,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "cfg": cfg,
    }
    torch.save(data, out)
    return None


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    #optimizer: torch.optim.Optimizer,
):

    data = torch.load(src)
    model.load_state_dict(data["model"])
    #optimizer.load_state_dict(data["optimizer"])
    return data["iteration"]
