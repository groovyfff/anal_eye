"""Verify promoted AE Brain runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path

CORE_RUNTIME_ARTIFACTS: tuple[str, ...] = (
    "features_schema.json",
    "tabular_model.joblib",
    "tabular_calibrator.joblib",
    "regime_model.joblib",
    "sequence_model.pt",
    "sequence_norm.npz",
    "rl_policy.zip",
    "meta_layer_mask.json",
)

META_MODE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "two_stage": ("meta_two_stage.joblib", "confidence_calibrator.joblib"),
    "legacy_3class": ("meta_model.joblib", "confidence_calibrator.joblib"),
    "side_specialists": (
        "long_specialist_model.joblib",
        "short_specialist_model.joblib",
        "confidence_calibrator_long.joblib",
        "confidence_calibrator_short.joblib",
        "side_configs.json",
        "training_regime.json",
    ),
    "side_aware_ensemble": (
        "meta_two_stage.joblib",
        "meta_model.joblib",
        "side_aware_ensemble.json",
        "confidence_calibrator_long.joblib",
        "confidence_calibrator_short.joblib",
    ),
}


def detect_meta_mode(artifacts_dir: Path) -> str:
    summary = artifacts_dir / "training_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        return str(data.get("meta", {}).get("meta_mode", "two_stage"))
    if (artifacts_dir / "side_specialists_report.json").exists():
        return "side_specialists"
    if (artifacts_dir / "side_aware_ensemble.json").exists():
        return "side_aware_ensemble"
    return "two_stage"


def required_runtime_artifacts(artifacts_dir: Path) -> list[str]:
    mode = detect_meta_mode(artifacts_dir)
    required = list(CORE_RUNTIME_ARTIFACTS)
    required.extend(META_MODE_ARTIFACTS.get(mode, META_MODE_ARTIFACTS["two_stage"]))
    return required


def missing_runtime_artifacts(artifacts_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in required_runtime_artifacts(artifacts_dir):
        if not (artifacts_dir / name).exists():
            missing.append(name)
    return missing


def verify_runtime_artifacts(artifacts_dir: Path) -> None:
    missing = missing_runtime_artifacts(artifacts_dir)
    if missing:
        raise FileNotFoundError(f"missing runtime artifacts in {artifacts_dir}: {', '.join(missing)}")
