"""Training regime filters applied consistently at train, inference, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ae_brain.training.side_configs import SideLabelConfigPair, load_side_configs


@dataclass(frozen=True, slots=True)
class TrainingRegimeConfig:
  min_vol_z: float | None = None
  apply_at_inference: bool = False

  def passes_vol_z(self, vol_z: float) -> bool:
    if self.min_vol_z is None:
      return True
    return float(vol_z) >= float(self.min_vol_z)

  def skip_reason_for(self, vol_z: float) -> str | None:
    if self.passes_vol_z(vol_z):
      return None
    return "below_min_vol_z"

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def save(self, artifacts_dir: Path) -> Path:
    path = artifacts_dir / "training_regime.json"
    path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    return path


def training_regime_from_side_configs(
  pair: SideLabelConfigPair,
  *,
  apply_at_inference: bool = False,
) -> TrainingRegimeConfig:
  return TrainingRegimeConfig(
    min_vol_z=pair.row_min_vol_z(),
    apply_at_inference=apply_at_inference,
  )


def load_training_regime(artifacts_dir: Path) -> TrainingRegimeConfig:
  path = artifacts_dir / "training_regime.json"
  if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainingRegimeConfig(
      min_vol_z=data.get("min_vol_z"),
      apply_at_inference=bool(data.get("apply_at_inference", False)),
    )
  pair = load_side_configs(artifacts_dir)
  return training_regime_from_side_configs(pair, apply_at_inference=False)


def regime_filter_skip_signal_components(reason: str, vol_z: float) -> dict[str, Any]:
  return {
    "decision_source": "training_regime_filter",
    "skip_reason": reason,
    "vol_z": round(float(vol_z), 6),
    "regime_filter": reason,
  }
