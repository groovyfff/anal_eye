"""Experiment tracking (JSON/CSV; optional MLflow)."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRun:
    run_id: str
    started_at: str
    git_commit: str = ""
    data_source: str = ""
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    date_range: dict[str, str] = field(default_factory=dict)
    label_config: dict[str, Any] = field(default_factory=dict)
    feature_schema: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    per_symbol_metrics: dict[str, Any] = field(default_factory=dict)
    decision_distribution: dict[str, int] = field(default_factory=dict)
    publishable_distribution: dict[str, int] = field(default_factory=dict)
    artifacts_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize config objects for experiment logs (dataclass, pydantic, dict, or plain object)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    legacy_dump = getattr(obj, "dict", None)
    if callable(legacy_dump):
        try:
            return legacy_dump()
        except TypeError:
            pass
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    raise TypeError(f"Cannot serialize config object of type {type(obj)!r}")


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ExperimentTracker:
    def __init__(self, reports_dir: Path, *, use_mlflow: bool = False) -> None:
        self._dir = reports_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._use_mlflow = use_mlflow
        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow

                self._mlflow = mlflow
                mlflow.set_tracking_uri(str(self._dir / "mlruns"))
            except ImportError:
                self._mlflow = None

    def start(self, **kwargs: Any) -> ExperimentRun:
        run = ExperimentRun(
            run_id=new_run_id(),
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit_hash(),
            **kwargs,
        )
        if self._mlflow is not None:
            self._mlflow.start_run(run_name=run.run_id)
        return run

    def log_run(self, run: ExperimentRun) -> Path:
        out = self._dir / f"run_{run.run_id}.json"
        out.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        csv_path = self._dir / "experiments.csv"
        row = {
            "run_id": run.run_id,
            "git_commit": run.git_commit,
            "symbols": ",".join(run.symbols),
            "test_ev": run.test_metrics.get("expected_ev_usd", ""),
            "test_pnl": run.test_metrics.get("net_pnl_usd", ""),
            "long_count": run.decision_distribution.get("LONG", 0),
            "short_count": run.decision_distribution.get("SHORT", 0),
            "artifacts": run.artifacts_path,
        }
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        if self._mlflow is not None:
            for key, val in row.items():
                if isinstance(val, (int, float)):
                    self._mlflow.log_metric(key, val)
            self._mlflow.end_run()
        return out
