"""Tests for train_multi_asset startup and config serialization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_brain.training.labels import LabelConfig
from ae_brain.training.tracking import ExperimentTracker, config_to_dict

_SPEC = importlib.util.spec_from_file_location(
    "train_multi_asset", ROOT / "scripts" / "train_multi_asset.py"
)
assert _SPEC and _SPEC.loader
tma = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tma
_SPEC.loader.exec_module(tma)


def test_config_to_dict_label_config_slots_dataclass() -> None:
    cfg = LabelConfig(tp_mult=2.0, sl_mult=1.0, horizon=12)
    dumped = config_to_dict(cfg)
    assert dumped == {
        "tp_mult": 2.0,
        "sl_mult": 1.0,
        "horizon": 12,
        "min_net_reward_usd": 1.0,
        "account_equity_usd": 100_000.0,
        "holding_hours": 8.0,
    }
    assert not hasattr(cfg, "__dict__")


def test_config_to_dict_supports_dict_and_plain_object() -> None:
    assert config_to_dict({"a": 1}) == {"a": 1}

    class Plain:
        def __init__(self) -> None:
            self.x = 1

    assert config_to_dict(Plain()) == {"x": 1}


def test_train_multi_asset_startup_without_crash(tmp_path: Path) -> None:
    dataset = tmp_path / "multi_asset.parquet"
    import pandas as pd

    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        for t in ts:
            rows.append(
                {
                    "timestamp": t,
                    "exchange": "binance",
                    "symbol": sym,
                    "timeframe": "1h",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "trades_count": 1,
                    "funding_rate": 0.0,
                    "open_interest": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(dataset)

    reports = tmp_path / "reports"
    out_dir = tmp_path / "artifacts"
    argv = [
        "train_multi_asset.py",
        "--dataset",
        str(dataset),
        "--symbols",
        "BTCUSDT,ETHUSDT",
        "--interval",
        "1h",
        "--reports-dir",
        str(reports),
        "--output-dir",
        str(out_dir),
        "--quick",
    ]

    with patch("scripts.train_production.main") as mock_main:
        with patch.object(sys, "argv", argv):
            tma.main()

    assert out_dir.exists()
    run_files = list(reports.glob("run_*.json"))
    assert run_files, "expected experiment run JSON"
    assert mock_main.called


def test_config_to_dict_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Cannot serialize"):
        config_to_dict(42)  # type: ignore[arg-type]
