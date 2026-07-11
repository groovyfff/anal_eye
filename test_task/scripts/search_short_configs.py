#!/usr/bin/env python3
"""Validation-only SHORT label config search (LONG config fixed as baseline)."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.config import Settings
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE
from ae_brain.training.metrics import brier
from ae_brain.training.side_configs import SideLabelConfig, SideLabelConfigPair
from ae_brain.training.specialist_dataset import collect_specialist_dataset
from ae_brain.training.specialist_metrics import precision_at_top_frac, simulate_publishable_ev
from ae_brain.training.specialist_train import train_specialist_pair
from scripts.train_production import (
    _build_layer_mask,
    _load_symbol_frames,
    fit_regime_and_features,
    train_rl_multi,
    train_sequence_multi,
    train_tabular_multi,
)


def _short_metrics_row(
    rep: dict,
    ds,
    short_cfg: SideLabelConfig,
    *,
    threshold: float = 0.70,
) -> dict:
    cut_train, cut_val = ds.train_val_cuts()
    F_val = ds.F[cut_train:cut_val]
    y_s = ds.y_short[cut_train:cut_val]
    ev_s = ds.ev_short[cut_train:cut_val]
    sym_val = ds.symbols[cut_train:cut_val]
    short_raw = np.array([rep["short_model"].predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)
    short_cal = np.array([rep["side_calibrators"].calibrate("SHORT", r) for r in short_raw], dtype=float)
    long_raw = np.array([rep["long_model"].predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)
    long_cal = np.array([rep["side_calibrators"].calibrate("LONG", r) for r in long_raw], dtype=float)
    pub = simulate_publishable_ev(long_cal, short_cal, ds.ev_long[cut_train:cut_val], ev_s, threshold=threshold)
    per_sym: dict[str, int] = {}
    for sym in sorted(set(sym_val)):
        m = sym_val == sym
        per_sym[str(sym)] = int(((short_cal[m] >= threshold) & (ev_s[m] > 0)).sum())
    return {
        "SHORT_config": short_cfg.to_dict(),
        "SHORT_positive_label_count": int(ds.y_short.sum()),
        "SHORT_auc": rep["short_metrics"].get("val_auc"),
        "SHORT_brier_raw": float(brier(y_s, short_raw)),
        "SHORT_brier_calibrated": float(brier(y_s, short_cal)),
        "SHORT_precision_top_1pct": precision_at_top_frac(y_s, short_cal, 0.01),
        "SHORT_precision_top_2pct": precision_at_top_frac(y_s, short_cal, 0.02),
        "SHORT_precision_top_5pct": precision_at_top_frac(y_s, short_cal, 0.05),
        "SHORT_precision_top_10pct": precision_at_top_frac(y_s, short_cal, 0.10),
        f"SHORT_publishable_ge_{threshold:.2f}": int(pub.get(f"publishable_SHORT_ge_{threshold:.2f}", 0)),
        "SHORT_publishable_EV": float(pub.get("publishable_EV_SHORT", 0.0)),
        "SHORT_publishable_per_symbol": per_sym,
        "validation_production_metrics": pub,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--base-artifacts", type=Path, default=None)
    parser.add_argument("--sample-per-symbol", type=int, default=15000)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--max-configs", type=int, default=0, help="0 = full grid")
    args = parser.parse_args()

    from scripts.train_multi_asset import _export_parquet_to_csv

    data_dir = ROOT / "data" / "cache" / "search_short"
    symbols = list(DEFAULT_SYMBOL_UNIVERSE)
    _export_parquet_to_csv(args.dataset, data_dir, symbols, "1h")
    frames = _load_symbol_frames(data_dir, symbols, "1h", sample_per_symbol=args.sample_per_symbol)

    settings = Settings()
    settings.model.artifacts_dir = args.base_artifacts or ROOT / "artifacts_candidates" / "20260702T192850Z"

    _, sym_data = fit_regime_and_features(frames, settings)
    tab = TabularPredictor(settings.model)
    tab.load(settings.model.artifacts_dir)
    training_summary = json.loads((settings.model.artifacts_dir / "training_summary.json").read_text())
    seq_m = training_summary.get("sequence", {})
    rl_m = training_summary.get("rl", {})
    import torch
    from ae_brain.layers.sequence import SequencePredictor
    from stable_baselines3 import PPO

    seq = SequencePredictor(settings.model, settings.gpu)
    seq.load(settings.model.artifacts_dir)
    seq_mod, seq_mean, seq_std, seq_dev = seq._module, seq._mean, seq._std, seq._device or torch.device("cpu")
    seq_win = settings.model.sequence_window
    pol = settings.model.artifacts_dir / "rl_policy.zip"
    rl_model = PPO.load(str(pol), device="auto") if pol.exists() else None
    layer_mask = _build_layer_mask(settings, seq_m, rl_m)

    long_baseline = SideLabelConfig(
        tp_mult=2.0, sl_mult=1.5, horizon=72, min_net_reward_usd=0.5, min_vol_z=-0.5
    )
    tp_mults = [1.0, 1.2, 1.5, 2.0]
    sl_mults = [1.0, 1.2, 1.5, 2.0]
    horizons = [12, 24, 48, 72]
    min_rewards = [0.25, 0.5, 1.0]
    min_vol_opts = [-0.5, -0.25, 0.0]

    grid = list(product(tp_mults, sl_mults, horizons, min_rewards, min_vol_opts))
    if args.max_configs and len(grid) > args.max_configs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grid), size=args.max_configs, replace=False)
        grid = [grid[i] for i in sorted(idx)]

    results = []
    best = None
    best_score = -1e18

    for tp, sl, h, mr, mv in grid:
        short_cfg = SideLabelConfig(tp_mult=tp, sl_mult=sl, horizon=h, min_net_reward_usd=mr, min_vol_z=mv)
        pair = SideLabelConfigPair(long=long_baseline, short=short_cfg)
        try:
            ds = collect_specialist_dataset(
                sym_data,
                settings,
                tab,
                seq_mod,
                seq_mean,
                seq_std,
                seq_dev,
                seq_win,
                rl_model,
                layer_mask,
                side_configs=pair,
                tb_horizon=max(long_baseline.horizon, short_cfg.horizon),
            )
            cut_train, cut_val = ds.train_val_cuts()
            rep = train_specialist_pair(
                ds.F,
                ds.y_long,
                ds.y_short,
                ds.ev_long,
                ds.ev_short,
                cut_train=cut_train,
                cut_val=cut_val,
                model_kind="lightgbm",
                calibration_method="sigmoid",
                symbols=ds.symbols,
                regime_ids=ds.regime_ids,
            )
            row = _short_metrics_row(rep, ds, short_cfg)
            score = float(row["SHORT_publishable_EV"]) + 1000 * int(row["SHORT_publishable_ge_0.70"])
            row["objective_score"] = score
            results.append(row)
            if score > best_score:
                best_score = score
                best = row
        except Exception as exc:
            results.append({"SHORT_config": short_cfg.to_dict(), "error": str(exc)})

    out = {
        "long_baseline": long_baseline.to_dict(),
        "n_configs": len(results),
        "best_SHORT_config": best,
        "all_results": results,
        "no_test_leakage": True,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / "short_config_search.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"best": best, "path": str(path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
