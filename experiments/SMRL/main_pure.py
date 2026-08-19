"""
Treinamento em PyTorch puro (sem Lightning).

Uso:
    python main_pure.py
    python main_pure.py --model proposed
    python main_pure.py --model gpt2 --config configs.yaml
"""

import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.baseline.models.gpt_model import GPTModel
from shared.data.dataset_loader import build_dataloaders, load_config
from experiments.cptoken.models.cptoken_model import CPTokenModel
from shared.tools.functions.loss import cross_entropy_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict, model_type: str) -> nn.Module:
    if model_type == "gpt2":
        return GPTModel(config)
    if model_type in ("proposed", "cptoken"):
        return CPTokenModel(config)
    raise ValueError(f"Modelo desconhecido: {model_type}")


def get_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        logits = model(input_ids)
        loss = cross_entropy_loss(logits, labels)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(config: dict, model_type: str) -> float:
    train_cfg = config["training"]
    log_cfg = config["logging"]
    seed = config.get("experiment", {}).get("seed", 42)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(
        tokenized_dataset_path=config["data"]["tokenized_dataset_path"],
        batch_size=train_cfg["batch_size"],
        num_workers=config["data"]["num_workers"],
        context_length=config["model"]["max_seq_length"],
        max_steps=train_cfg["max_steps"],
    )

    model = build_model(config, model_type).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {model_type} | Parâmetros treináveis: {n_params:,} | Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg.get("warm_up_steps", 0)
    grad_clip = train_cfg.get("gradient_clip_val", 0.5)
    logging_steps = log_cfg.get("logging_steps", 50)

    train_loader = dataloaders["train"]
    valid_loader = dataloaders.get("valid")

    model.train()
    running_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        lr = get_lr(step, train_cfg["lr"], warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, labels)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        running_loss += loss.item()

        if (step + 1) % logging_steps == 0:
            avg_loss = running_loss / logging_steps
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(
                f"step {step + 1:>5}/{max_steps} | "
                f"train_loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{steps_per_sec:.2f} it/s"
            )
            running_loss = 0.0

    if valid_loader is not None:
        test_loss = evaluate(model, valid_loader, device)
        print(f"test_loss {test_loss:.4f}")
        return test_loss

    return math.nan


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento CPToken/GPT em PyTorch puro")
    parser.add_argument(
        "--config",
        default="experiments/cptoken/configs/default.yaml",
        help="Caminho do arquivo de config",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["gpt2", "proposed", "cptoken"],
        help="Tipo de modelo (padrão: model.type do config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    model_type = args.model or cfg["model"].get("type", "proposed")
    train(cfg, model_type)
