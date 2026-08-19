from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from shared.data.dataset_loader import load_config
from experiments.cptoken.main import train
from shared.tools.utils.sweep import RunSpec, load_sweep_runs


def _format_overrides(overrides: dict) -> str:
    if not overrides:
        return "(base config)"
    return ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))


def _extract_test_loss(result) -> float | None:
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get("test_loss")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("test_loss")
    return None


def _run_to_summary(run_spec: RunSpec, status: str, result=None, error: str | None = None) -> dict:
    return {
        "run_index": run_spec.run_index,
        "run_name": run_spec.run_name,
        "model_name": run_spec.model_name,
        "status": status,
        "overrides": run_spec.overrides,
        "test_loss": _extract_test_loss(result),
        "n_params": result.get("n_params") if isinstance(result, dict) else None,
        "error": error,
    }


def _save_summary(sweep_name: str, results_dir: Path, summaries: list[dict]) -> Path:
    output_dir = results_dir / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sweep_name}.json"
    payload = {
        "sweep_name": sweep_name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runs": summaries,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _print_run_plan(run_specs: list[RunSpec]) -> None:
    print(f"Planned runs: {len(run_specs)}")
    for run_spec in run_specs:
        print(
            f"  [{run_spec.run_index + 1}/{len(run_specs)}] "
            f"{run_spec.run_name} | model={run_spec.model_name} | {_format_overrides(run_spec.overrides)}"
        )


def run_sweep(sweep_path: str, start: int = 0, dry_run: bool = False, results_dir: str = "results/cptoken/") -> int:
    run_specs = load_sweep_runs(sweep_path, load_config)
    sweep_name = run_specs[0].sweep_name

    if start < 0 or start >= len(run_specs):
        raise ValueError(f"--start must be between 0 and {len(run_specs) - 1}.")

    _print_run_plan(run_specs)

    if dry_run:
        print("Dry run enabled. No training was started.")
        return 0

    summaries: list[dict] = []
    selected_runs = run_specs[start:]

    for offset, run_spec in enumerate(selected_runs, start=start):
        position = offset + 1
        total = len(run_specs)
        print(
            f"\n[{position}/{total}] Starting {run_spec.run_name} "
            f"(model={run_spec.model_name}, {_format_overrides(run_spec.overrides)})"
        )

        try:
            result = train(
                run_spec.config,
                run_spec.model_name,
                sweep_name=run_spec.sweep_name,
                run_name=run_spec.run_name,
                run_index=run_spec.run_index,
            )
            summary = _run_to_summary(run_spec, status="success", result=result)
            print(f"[{position}/{total}] Finished {run_spec.run_name} | test_loss={summary['test_loss']}")
        except Exception as exc:
            summary = _run_to_summary(
                run_spec,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[{position}/{total}] Failed {run_spec.run_name}: {summary['error']}")
            traceback.print_exc()

        summaries.append(summary)

    output_path = _save_summary(sweep_name, Path(results_dir), summaries)
    print(f"\nSweep summary saved to {output_path}")
    failed = sum(1 for item in summaries if item["status"] == "failed")
    print(f"Completed with {len(summaries) - failed} success(es) and {failed} failure(s).")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a serial hyperparameter sweep.")
    parser.add_argument("--sweep", required=True, help="Path to sweep YAML file.")
    parser.add_argument("--start", type=int, default=0, help="Run index to start from.")
    parser.add_argument("--dry-run", action="store_true", help="List planned runs without training.")
    parser.add_argument(
        "--results-dir",
        default="results/cptoken/",
        help="Directory for sweep summaries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_sweep(args.sweep, start=args.start, dry_run=args.dry_run, results_dir=args.results_dir))
