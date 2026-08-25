"""Sweep classification runs by mutating copies of the dict from load_config."""

from __future__ import annotations

import argparse
import itertools

from experiments.SMRL.main_cl import train
from shared.data.dataset_loader import load_config
from shared.tools.utils.sweep import apply_overrides


PARAMETERS = {
    "model.type": ["SMRL"],
    "experiment.seed": [42, 123, 7],
    "model.kind": ['dct', 'dst', 'haar', 'dht', 'identity', 'learnable']
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SMRL classification configs in-memory.")
    parser.add_argument(
        "--config",
        default="experiments/SMRL/configs/default.yaml",
        help="Base YAML loaded once via load_config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)

    keys = list(PARAMETERS)
    for combo in itertools.product(*(PARAMETERS[k] for k in keys)):
        print('Configuration: ', "_".join(f"{key.split('.')[-1]}={value}" for key, value in combo))
        overrides = dict(zip(keys, combo))
        config = apply_overrides(base_config, overrides)
        config["experiment"]["name"] = "_".join(
            f"{key.split('.')[-1]}={value}" for key, value in overrides.items()
        )
        print(f"Running {config['experiment']['name']}", flush=True)
        train(config, config["model"]["type"])


if __name__ == "__main__":
    main()
