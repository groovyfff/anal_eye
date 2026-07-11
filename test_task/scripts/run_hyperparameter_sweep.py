#!/usr/bin/env python3
"""Optuna hyperparameter sweep (optional dependency)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "production")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Install with: pip install optuna", file=sys.stderr)
        print("Or run baseline: python scripts/train_multi_asset.py --symbols-from-config", file=sys.stderr)
        sys.exit(1)

    def objective(trial: "optuna.Trial") -> float:
        tp_mult = trial.suggest_float("tp_mult", 1.5, 3.5)
        sl_mult = trial.suggest_float("sl_mult", 1.0, 2.5)
        horizon = trial.suggest_int("horizon", 12, 48)
        min_conviction = trial.suggest_float("min_conviction", 0.45, 0.65)
        run_id = trial.number
        out = ROOT / "artifacts_candidates" / f"sweep_{run_id}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_multi_asset.py"),
            "--data-dir",
            str(args.data_dir),
            "--symbols-from-config",
            "--output-dir",
            str(out),
            "--tp-mult",
            str(tp_mult),
            "--sl-mult",
            str(sl_mult),
            "--horizon",
            str(horizon),
            "--min-conviction",
            str(min_conviction),
            "--quick",
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        metrics_path = out / "test_metrics.json"
        if not metrics_path.exists():
            return -1e9
        metrics = json.loads(metrics_path.read_text())
        return float(metrics.get("expected_ev_usd", -1e9))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "optuna_best.json").write_text(
        json.dumps({"best_params": study.best_params, "best_value": study.best_value}, indent=2)
    )
    print(json.dumps({"best_params": study.best_params, "best_value": study.best_value}, indent=2))


if __name__ == "__main__":
    main()
