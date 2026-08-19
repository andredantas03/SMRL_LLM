import argparse, json, os, time, copy, random
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch
import yaml


# --- Config tipada com validação básica (stdlib) ---
@dataclass
class ModelCfg:
    d_model: int
    num_layers: int
    num_heads: int
    vocab_size: int
    max_seq_len: int


@dataclass
class TrainCfg:
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    seed: int
    mixed_precision: bool


@dataclass
class DataCfg:
    path: str
    num_workers: int


@dataclass
class RunCfg:
    output_dir: str


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], keys):
    for k in keys:
        d = d[k]
    return d


def deep_set(d: Dict[str, Any], keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_overrides(kvs):
    """
    Overrides no formato: model.d_model=512 train.lr=1e-4
    """
    changes = []
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"Override inválido: {kv} (use chave=valor)")
        k, v = kv.split("=", 1)
        path = k.split(".")
        # tenta converter para número/bool automaticamente
        if v.lower() in ("true", "false"):
            val = v.lower() == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        changes.append((path, val))
    return changes


def apply_overrides(cfg: Dict[str, Any], overrides):
    cfg = copy.deepcopy(cfg)
    for path, val in overrides:
        deep_set(cfg, path, val)
    return cfg


def validate_cfg(cfg: Dict[str, Any]):
    # Validações simples (suficiente para iniciante)
    assert cfg["model"]["d_model"] > 0
    assert cfg["model"]["n_head"] > 0
    assert cfg["train"]["lr"] > 0
    assert cfg["train"]["batch_size"] > 0
    assert cfg["train"]["max_steps"] > 0


def make_run_dir(base_dir: str) -> str:
    ts = 'best_model'
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: str, cfg: Dict[str, Any]):
    path = os.path.join(run_dir, "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser():
    # build_argparser() devolve um argparse.ArgumentParser já configurado (com opções como --config e --override).
    ap = argparse.ArgumentParser(description="Training loop com config externa")
    ap.add_argument(
        "--config", default="C:\\Transformers\\kron_llms\\configs\\base.yaml")
    ap.add_argument(
        "--override", nargs="*", default=[], help="ex: model.d_model=512 train.lr=1e-4"
    )
    # nargs=...
    # Controla quantos valores aquela flag consome.
    # nargs="*" → zero ou mais valores.
    # Por isso --override aceita nenhum, um ou vários itens:
    return ap
