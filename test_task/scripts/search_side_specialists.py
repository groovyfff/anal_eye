#!/usr/bin/env python3
"""Validation-only label/model search for side specialists (no test leakage)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.config import Settings
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE
from ae_brain.training.labels import LabelConfig
from ae_brain.training.specialist_dataset import collect_specialist_dataset
from ae_brain.training.specialist_train import train_specialist_pair
from scripts.train_production import (
    _build_layer_mask,
    _load_symbol_frames,
    fit_regime_and_features,
    train_rl_multi,
    train_sequence_multi,
    train_tabular_multi,
)


def _prepare_base_stack(sym_data, settings, *, seq_epochs: int, rl_steps: int, retrain: bool):
    if retrain:
        tab_m = train_tabular_multi(sym_data, settings)
        seq_m, seq_mod, seq_mean, seq_std, seq_dev, seq_win = train_sequence_multi(
            sym_data, settings, epochs=seq_epochs, cap=60000, batch_size=256
        )
        rl_m, rl_model = train_rl_multi(sym_data, settings, total_timesteps=rl_steps)
    else:
        tab_m = {}
        seq_m = json.loads((settings.model.artifacts_dir / "training_summary.json").read_text()).get("sequence", {})
        rl_m = json.loads((settings.model.artifacts_dir / "training_summary.json").read_text()).get("rl", {})
        import torch
        from ae_brain.layers.sequence import SequencePredictor
        from stable_baselines3 import PPO

        seq = SequencePredictor(settings.model, settings.gpu)
        seq.load(settings.model.artifacts_dir)
        seq_mod, seq_mean, seq_std, seq_dev = seq._module, seq._mean, seq._std, seq._device or torch.device("cpu")
        seq_win = settings.model.sequence_window
        pol = settings.model.artifacts_dir / "rl_policy.zip"
        rl_model = PPO.load(str(pol), device="auto") if pol.exists() else None

    tab = TabularPredictor(settings.model)
    tab.load(settings.model.artifacts_dir)
    layer_mask = _build_layer_mask(settings, seq_m, rl_m)
    return tab, seq_mod, seq_mean, seq_std, seq_dev, seq_win, rl_model, layer_mask, seq_m, rl_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--base-artifacts", type=Path, default=None, help="Reuse tabular/seq/rl from prior run")
    parser.add_argument("--sample-per-symbol", type=int, default=15000)
    parser.add_argument("--seq-epochs", type=int, default=2)
    parser.add_argument("--rl-timesteps", type=int, default=3000)
    parser.add_argument("--max-configs", type=int, default=24)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    data_dir = ROOT / "data" / "cache" / "search_specialists"
    from scripts.train_multi_asset import _export_parquet_to_csv

    symbols = list(DEFAULT_SYMBOL_UNIVERSE)
    _export_parquet_to_csv(args.dataset, data_dir, symbols, "1h")
    frames = _load_symbol_frames(data_dir, symbols, "1h", sample_per_symbol=args.sample_per_symbol)

    settings = Settings()
    if args.base_artifacts:
        settings.model.artifacts_dir = args.base_artifacts
    else:
        out = ROOT / "artifacts_candidates" / "search_base_tmp"
        out.mkdir(parents=True, exist_ok=True)
        settings.model.artifacts_dir = out

    _, sym_data = fit_regime_and_features(frames, settings)
    tab, seq_mod, seq_mean, seq_std, seq_dev, seq_win, rl_model, layer_mask, seq_m, rl_m = _prepare_base_stack(
        sym_data, settings, seq_epochs=args.seq_epochs, rl_steps=args.rl_timesteps, retrain=args.base_artifacts is None
    )

    horizons = [12, 24, 48, 72]
    tp_mults = [2.0, 2.5, 3.0]
    sl_mults = [1.25, 1.5, 2.0]
    min_rewards = [0.5, 1.0, 2.0]
    min_vol_opts: list[float | None] = [None, -0.5]
    model_kinds = ["logreg", "lightgbm"]
    cal_methods = ["isotonic", "sigmoid"]

    grid = list(product(horizons, tp_mults, sl_mults, min_rewards, min_vol_opts, model_kinds, cal_methods))
    if len(grid) > args.max_configs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grid), size=args.max_configs, replace=False)
        grid = [grid[i] for i in sorted(idx)]

    results = []
    best = None
    best_score = -1e18

    for h, tp, sl, mr, mv, mk, cm in grid:
        label_cfg = LabelConfig(tp_mult=tp, sl_mult=sl, horizon=h, min_net_reward_usd=mr)
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
                label_cfg=label_cfg,
                min_vol_z=mv,
                use_extended_features=True,
                tb_horizon=h,
            )
        except Exception as exc:
            results.append({"error": str(exc), "label_config": asdict(label_cfg)})
            continue
        if len(ds) < 500:
            continue
        cut_train, cut_val = ds.train_val_cuts()
        try:
            rep = train_specialist_pair(
                ds.F,
                ds.y_long,
                ds.y_short,
                ds.ev_long,
                ds.ev_short,
                cut_train=cut_train,
                cut_val=cut_val,
                model_kind=mk,
                calibration_method=cm,
                symbols=ds.symbols,
                regime_ids=ds.regime_ids,
            )
        except Exception as exc:
            results.append({"error": str(exc), "label_config": ds.label_config})
            continue
        prod = rep["validation_production_metrics"]
        pub_ev = float(prod.get("publishable_EV_total", 0.0))
        pub_l = int(prod.get("publishable_LONG_ge_0.70", 0))
        pub_s = int(prod.get("publishable_SHORT_ge_0.70", 0))
        score = pub_ev
        if pub_l == 0 or pub_s == 0:
            score -= 1e6
        row = {
            "label_config": ds.label_config,
            "model_kind": mk,
            "calibration_method": cm,
            "validation_production_metrics": prod,
            "long_auc": rep["long_metrics"].get("val_auc"),
            "short_auc": rep["short_metrics"].get("val_auc"),
            "LONG_ceiling": rep["confidence_ceiling"]["LONG"].get("ceiling_diagnosis"),
            "SHORT_ceiling": rep["confidence_ceiling"]["SHORT"].get("ceiling_diagnosis"),
            "LONG_top5_raw_precision": prod.get("precision_at_top_5pct_LONG_raw"),
            "SHORT_top5_raw_precision": prod.get("precision_at_top_5pct_SHORT_raw"),
            "objective_score": score,
        }
        results.append(row)
        if score > best_score:
            best_score = score
            best = row

    out = {
        "n_configs_tried": len(results),
        "best": best,
        "all_results": sorted(results, key=lambda r: r.get("objective_score", -1e18), reverse=True)[:10],
        "no_test_leakage": True,
        "note": "Objective = validation publishable_EV_total; penalty if either side publishable count is 0",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / "side_specialist_search.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
