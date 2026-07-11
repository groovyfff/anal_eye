#!/usr/bin/env python3
"""Diagnose side-specialist confidence ceiling on validation slice (no test leakage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.config import Settings
from ae_brain.layers.risk_agent import RiskAgent
from ae_brain.layers.sequence import SequencePredictor
from ae_brain.layers.side_specialists import SideSpecialistModel
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE
from ae_brain.training.calibration import SideCalibrators
from ae_brain.training.labels import LabelConfig
from ae_brain.training.specialist_dataset import collect_specialist_dataset
from ae_brain.training.specialist_metrics import confidence_ceiling_report, simulate_publishable_ev
from scripts.train_multi_asset import _export_parquet_to_csv
from scripts.train_production import _load_symbol_frames, fit_regime_and_features
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--sample-per-symbol", type=int, default=15000)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    artifacts = args.artifacts or (ROOT / "artifacts_candidates" / args.run_id)
    report_path = artifacts / "side_specialists_report.json"
    if not report_path.exists():
        print(f"Missing {report_path}", file=sys.stderr)
        sys.exit(1)
    train_report = json.loads(report_path.read_text(encoding="utf-8"))
    splits = train_report.get("splits", {})
    cut_train = int(splits.get("n_train", int(89256 * 0.70)))
    n_val = int(splits.get("n_validation", int(89256 * 0.15)))
    cut_val = cut_train + n_val

    data_dir = ROOT / "data" / "cache" / "diagnose_parquet"
    symbols = list(DEFAULT_SYMBOL_UNIVERSE)
    _export_parquet_to_csv(args.dataset, data_dir, symbols, "1h")
    frames = _load_symbol_frames(data_dir, symbols, "1h", sample_per_symbol=args.sample_per_symbol)

    settings = Settings()
    settings.model.artifacts_dir = artifacts
    _, sym_data = fit_regime_and_features(frames, settings)

    tab = TabularPredictor(settings.model)
    tab.load(artifacts)
    layer_mask = json.loads((artifacts / "meta_layer_mask.json").read_text(encoding="utf-8"))

    seq = SequencePredictor(settings.model, settings.gpu)
    seq.load(artifacts)
    rl_path = artifacts / "rl_policy.zip"
    rl_model = PPO.load(str(rl_path), device="auto") if rl_path.exists() else None

    label_cfg = LabelConfig(tp_mult=2.5, sl_mult=1.5, horizon=24)

    ds = collect_specialist_dataset(
        sym_data,
        settings,
        tab,
        seq._module,
        seq._mean,
        seq._std,
        seq._device or torch.device("cpu"),
        settings.model.sequence_window,
        rl_model,
        layer_mask,
        label_cfg=label_cfg,
        use_extended_features=False,
    )

    long_m = SideSpecialistModel("LONG").load(artifacts)
    short_m = SideSpecialistModel("SHORT").load(artifacts)
    cals = SideCalibrators(settings.model.calibration_method).load(artifacts)

    F_val = ds.F[cut_train:cut_val]
    y_l = ds.y_long[cut_train:cut_val]
    y_s = ds.y_short[cut_train:cut_val]
    ev_l = ds.ev_long[cut_train:cut_val]
    ev_s = ds.ev_short[cut_train:cut_val]
    sym_v = ds.symbols[cut_train:cut_val]
    reg_v = ds.regime_ids[cut_train:cut_val]

    long_raw = np.array([long_m.predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)
    short_raw = np.array([short_m.predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)
    long_cal = np.array([cals.calibrate("LONG", r) for r in long_raw], dtype=float)
    short_cal = np.array([cals.calibrate("SHORT", r) for r in short_raw], dtype=float)

    out = {
        "run_id": args.run_id,
        "slice": "validation_calibration_only",
        "no_test_leakage": True,
        "LONG": confidence_ceiling_report(
            side="LONG", y_true=y_l, raw=long_raw, calibrated=long_cal, ev_usd=ev_l, symbols=sym_v, regime_ids=reg_v
        ),
        "SHORT": confidence_ceiling_report(
            side="SHORT", y_true=y_s, raw=short_raw, calibrated=short_cal, ev_usd=ev_s, symbols=sym_v, regime_ids=reg_v
        ),
        "publishable_simulation": simulate_publishable_ev(long_cal, short_cal, ev_l, ev_s),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / f"confidence_ceiling_{args.run_id}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
