"""Per-side label configuration for independent LONG/SHORT specialist training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ae_brain.training.labels import LabelConfig


@dataclass(frozen=True, slots=True)
class SideLabelConfig:
  tp_mult: float = 2.0
  sl_mult: float = 1.5
  horizon: int = 72
  min_net_reward_usd: float = 0.5
  min_vol_z: float | None = -0.5

  def to_label_config(self) -> LabelConfig:
    return LabelConfig(
      tp_mult=self.tp_mult,
      sl_mult=self.sl_mult,
      horizon=self.horizon,
      min_net_reward_usd=self.min_net_reward_usd,
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class SideLabelConfigPair:
  long: SideLabelConfig
  short: SideLabelConfig

  @staticmethod
  def default_baseline() -> "SideLabelConfigPair":
    cfg = SideLabelConfig(
      tp_mult=2.0,
      sl_mult=1.5,
      horizon=72,
      min_net_reward_usd=0.5,
      min_vol_z=-0.5,
    )
    return SideLabelConfigPair(long=cfg, short=cfg)

  @staticmethod
  def from_dict(data: dict[str, Any]) -> "SideLabelConfigPair":
    if "long" in data and "short" in data:
      return SideLabelConfigPair(
        long=SideLabelConfig(**{k: data["long"][k] for k in SideLabelConfig.__dataclass_fields__ if k in data["long"]}),
        short=SideLabelConfig(**{k: data["short"][k] for k in SideLabelConfig.__dataclass_fields__ if k in data["short"]}),
      )
    # Legacy single-config format -> apply to both sides.
    shared = SideLabelConfig(
      tp_mult=float(data.get("tp_mult", 2.0)),
      sl_mult=float(data.get("sl_mult", 1.5)),
      horizon=int(data.get("horizon", 72)),
      min_net_reward_usd=float(data.get("min_net_reward_usd", 0.5)),
      min_vol_z=data.get("min_vol_z"),
    )
    return SideLabelConfigPair(long=shared, short=shared)

  def row_min_vol_z(self) -> float | None:
    """Most permissive vol_z gate for dataset row inclusion."""
    vals = [v for v in (self.long.min_vol_z, self.short.min_vol_z) if v is not None]
    return min(vals) if vals else None

  def max_horizon(self) -> int:
    return max(self.long.horizon, self.short.horizon)

  def to_dict(self) -> dict[str, Any]:
    return {"long": self.long.to_dict(), "short": self.short.to_dict()}

  def save(self, artifacts_dir: Path) -> Path:
    path = artifacts_dir / "side_configs.json"
    path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    return path


def load_side_configs(artifacts_dir: Path) -> SideLabelConfigPair:
  path = artifacts_dir / "side_configs.json"
  if path.exists():
    return SideLabelConfigPair.from_dict(json.loads(path.read_text(encoding="utf-8")))
  report = artifacts_dir / "side_specialists_report.json"
  if report.exists():
    rep = json.loads(report.read_text(encoding="utf-8"))
    lc = rep.get("label_config") or rep.get("side_configs")
    if lc:
      return SideLabelConfigPair.from_dict(lc if "long" in lc else lc)
  summary = artifacts_dir / "training_summary.json"
  if summary.exists():
    meta = json.loads(summary.read_text(encoding="utf-8")).get("meta", {})
    lc = meta.get("side_configs") or meta.get("label_config")
    if lc:
      return SideLabelConfigPair.from_dict(lc if "long" in lc else lc)
  return SideLabelConfigPair.default_baseline()
