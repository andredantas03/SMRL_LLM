from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SWEEP_TOP_LEVEL_KEYS = {
    "name",
    "base_config",
    "model_name",
    "mode",
    "parameters",
    "runs",
    "max_runs",
}


@dataclass
class RunSpec:
    config: dict
    run_name: str
    run_index: int
    sweep_name: str
    model_name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepSpec:
    name: str
    base_config_path: str
    model_name: str
    mode: str
    max_runs: int
    parameters: dict[str, list[Any]]
    runs: list[dict[str, Any]]


def deep_copy_config(config: dict) -> dict:
    return copy.deepcopy(config)


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    updated = deep_copy_config(config)
    for key, value in overrides.items():
        set_nested(updated, key, value)
    return updated


def _format_override_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.0e}" if value < 0.01 or value >= 1000 else str(value)
        return text.replace(".", "p").replace("-", "m")
    return str(value).replace(".", "p").replace(" ", "_")


def build_run_name(overrides: dict[str, Any], run_index: int, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name

    if not overrides:
        return f"run_{run_index:03d}"

    parts = []
    for key in sorted(overrides):
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}{_format_override_value(overrides[key])}")
    return "_".join(parts)


def generate_grid_runs(
    base_config: dict,
    parameters: dict[str, list[Any]],
    sweep_name: str,
    model_name: str,
) -> list[RunSpec]:
    if not parameters:
        raise ValueError("Grid sweep requires at least one parameter.")

    keys = list(parameters.keys())
    value_lists = [parameters[key] for key in keys]
    run_specs: list[RunSpec] = []

    for run_index, combo in enumerate(itertools.product(*value_lists)):
        overrides = dict(zip(keys, combo))
        config = apply_overrides(base_config, overrides)
        run_specs.append(
            RunSpec(
                config=config,
                run_name=build_run_name(overrides, run_index),
                run_index=run_index,
                sweep_name=sweep_name,
                model_name=model_name,
                overrides=overrides,
            )
        )

    return run_specs


def generate_explicit_runs(
    base_config: dict,
    runs: list[dict[str, Any]],
    sweep_name: str,
    model_name: str,
) -> list[RunSpec]:
    if not runs:
        raise ValueError("Explicit sweep requires at least one run.")

    run_specs: list[RunSpec] = []
    for run_index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Run at index {run_index} must be a mapping.")

        overrides = run.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Run at index {run_index} has invalid overrides.")

        run_model_name = run.get("model_name", model_name)
        config = apply_overrides(base_config, overrides)
        run_specs.append(
            RunSpec(
                config=config,
                run_name=build_run_name(overrides, run_index, explicit_name=run.get("name")),
                run_index=run_index,
                sweep_name=sweep_name,
                model_name=run_model_name,
                overrides=overrides,
            )
        )

    return run_specs


def load_sweep_spec(path: str | Path) -> SweepSpec:
    sweep_path = Path(path)
    with sweep_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Sweep file must contain a YAML mapping.")

    unknown_keys = set(raw) - SWEEP_TOP_LEVEL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown sweep keys: {sorted(unknown_keys)}")

    name = raw.get("name")
    if not name:
        raise ValueError("Sweep file must define 'name'.")

    model_name = raw.get("model_name")
    if not model_name:
        raise ValueError("Sweep file must define 'model_name'.")

    mode = raw.get("mode", "grid")
    if mode not in {"grid", "explicit"}:
        raise ValueError("Sweep mode must be 'grid' or 'explicit'.")

    return SweepSpec(
        name=name,
        base_config_path=raw.get("base_config", "configs.yaml"),
        model_name=model_name,
        mode=mode,
        max_runs=int(raw.get("max_runs", 50)),
        parameters=raw.get("parameters") or {},
        runs=raw.get("runs") or [],
    )


def build_run_configs(spec: SweepSpec, base_config: dict) -> list[RunSpec]:
    if spec.mode == "grid":
        run_specs = generate_grid_runs(
            base_config=base_config,
            parameters=spec.parameters,
            sweep_name=spec.name,
            model_name=spec.model_name,
        )
    else:
        run_specs = generate_explicit_runs(
            base_config=base_config,
            runs=spec.runs,
            sweep_name=spec.name,
            model_name=spec.model_name,
        )

    if not run_specs:
        raise ValueError("Sweep produced zero runs.")

    if len(run_specs) > spec.max_runs:
        raise ValueError(
            f"Sweep '{spec.name}' would produce {len(run_specs)} runs, "
            f"which exceeds max_runs={spec.max_runs}."
        )

    if len(run_specs) == spec.max_runs:
        warnings.warn(
            f"Sweep '{spec.name}' reached max_runs={spec.max_runs}. "
            "Increase max_runs if more combinations are expected.",
            stacklevel=2,
        )

    return run_specs


def load_sweep_runs(sweep_path: str | Path, config_loader) -> list[RunSpec]:
    spec = load_sweep_spec(sweep_path)
    base_config = config_loader(spec.base_config_path)
    return build_run_configs(spec, base_config)
