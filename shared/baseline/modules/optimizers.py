import torch
from collections.abc import Callable
from typing import Optional
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class Adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p.data, requires_grad=False)
                state["v"] = torch.zeros_like(p.data, requires_grad=False)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state associated with p.
                state = self.state[p]

                # Get iteration number from the state, or initial value.
                t = state.get("t", 1)

                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                # (Update the first moment estimate)
                state["m"] = state["m"] * beta1 + (1 - beta1) * grad

                # (Update the second moment estimate)
                state["v"] = state["v"] * beta2 + (1 - beta2) * grad.pow(2)

                # (Compute adjusted α for iteration t)
                lr_t = lr * (math.sqrt((1 - beta2**t))) / (1 - beta1**t)

                # (Update the parameters)

                p.data = p.data - lr_t * state["m"] / (
                    torch.sqrt(state["v"]) + eps
                )  # Update weight tensor in-place.

                # (Apply weight decay)
                p.data = p.data - lr * weight_decay * p.data

                # Increment iteration number.
                state["t"] = t + 1

        return loss
