#!/usr/bin/env python3
"""Multi-asset training orchestrator with EV-aware labels and walk-forward evaluation.

Wraps the production ensemble trainer, writes artifacts to artifacts_candidates/<run_id>/,
and never overwrites test_task/artifacts/ unless explicitly promoted.

Usage::

    python scripts/train_multi_asset.py --symbols-from-config --data-dir data/production
    python scripts/train_multi_asset.py --dataset data/datasets/multi_asset.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, default_allowed_symbols_csv, parse_symbol_list
from ae_brain.training.labels import LabelConfig, label_distribution_report
from ae_brain.training.side_configs import SideLabelConfig, SideLabelConfigPair
from ae_brain.training.tracking import ExperimentTracker, config_to_dict


def _export_parquet_to_csv(dataset: Path, out_dir: Path, symbols: list[str], interval: str) -> Path:
    import pandas as pd

    from ae_brain.training.canonical import candles_from_canonical

    df = pd.read_parquet(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        sub = df[(df["symbol"] == sym) & (df["timeframe"] == interval)]
        if sub.empty:
            continue
        candles = candles_from_canonical(sub)
        path = out_dir / f"{sym}_{interval}.csv"
        candles.to_csv(path, index=False)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "production")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--tp-mult", type=float, default=2.5)
    parser.add_argument("--sl-mult", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--min-net-reward", type=float, default=None, help="min_net_reward_usd for EV-aware labels")
    parser.add_argument("--min-vol-z", type=float, default=None, help="Skip rows with vol_z below this (noise filter)")
    parser.add_argument("--side-configs", type=Path, default=None, help="JSON with separate long/short label configs")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation during specialist training")
    parser.add_argument(
        "--apply-regime-filter-at-inference",
        action="store_true",
        help="Apply training min_vol_z filter at inference (SKIP below_min_vol_z)",
    )
    parser.add_argument("--short-search", action="store_true", help="Run validation-only SHORT label config search after training")
    parser.add_argument("--multi-timeframe", default="", help="Comma-separated auxiliary timeframes (e.g. 15m)")
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"], default=None)
    parser.add_argument("--specialist-model-kind", choices=["logreg", "lightgbm"], default="lightgbm")
    parser.add_argument("--min-conviction", type=float, default=None)
    parser.add_argument("--seq-epochs", type=int, default=4)
    parser.add_argument("--rl-timesteps", type=int, default=150_000)
    parser.add_argument("--quick", action="store_true", help="Reduced RL/sequence budget for sweeps")
    parser.add_argument("--medium", action="store_true", help="Memory-safe medium run: sample rows, low seq/RL budget")
    parser.add_argument("--sample-per-symbol", type=int, default=None)
    parser.add_argument("--meta-mode", choices=["two_stage", "legacy_3class", "side_aware_ensemble", "side_specialists"], default="two_stage")
    parser.add_argument("--balance-side-specialists", default="false",
                        choices=["true", "false"],
                        help="Enable class balancing for side specialists (scale_pos_weight / class_weight)")
    parser.add_argument("--long-positive-weight", default="auto",
                        help="LONG profitable-class weight: 'auto' (neg/pos imbalance) or a float")
    parser.add_argument("--short-positive-weight", default="auto",
                        help="SHORT profitable-class weight: 'auto' (neg/pos imbalance) or a float")
    parser.add_argument("--balance-train-samples", default="false",
                        choices=["true", "false"],
                        help="Balance (undersample) the majority class within the train split only")
    parser.add_argument("--max-side-train-samples-per-class", type=int, default=None,
                        help="Cap samples per class when balance-train-samples is enabled")
    parser.add_argument("--allow-skip-sequence", default="false", choices=["true", "false"],
                        help="Memory-safe: skip sequence training if it is too heavy or fails")
    parser.add_argument("--skip-sequence", default="false", choices=["true", "false"],
                        help="Proactively skip PatchTST sequence training (avoids OOM/SIGKILL)")
    parser.add_argument("--skip-rl", default="false", choices=["true", "false"],
                        help="Proactively skip PPO RL risk-agent training")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--use-mlflow", action="store_true")
    args = parser.parse_args()

    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    data_dir = args.data_dir
    if args.dataset:
        data_dir = ROOT / "data" / "cache" / "from_parquet"
        _export_parquet_to_csv(args.dataset, data_dir, symbols, args.interval)

    tracker = ExperimentTracker(args.reports_dir, use_mlflow=args.use_mlflow)

    if args.side_configs and args.side_configs.exists():
        side_pair = SideLabelConfigPair.from_dict(json.loads(args.side_configs.read_text(encoding="utf-8")))
    else:
        long_cfg = SideLabelConfig(
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
            horizon=args.horizon,
            min_net_reward_usd=args.min_net_reward if args.min_net_reward is not None else 0.5,
            min_vol_z=args.min_vol_z,
        )
        side_pair = SideLabelConfigPair(long=long_cfg, short=long_cfg)

    label_config = side_pair.long.to_label_config()
    run = tracker.start(
        symbols=symbols,
        timeframes=[args.interval],
        label_config={**config_to_dict(label_config), "side_configs": side_pair.to_dict()},
    )
    out_dir = args.output_dir or (ROOT / "artifacts_candidates" / run.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["AEB_MODEL_ARTIFACTS_DIR"] = str(out_dir)
    if args.min_conviction is not None:
        os.environ["AEB_FUSION_MIN_CONVICTION"] = str(args.min_conviction)
    if args.calibration_method:
        os.environ["AEB_MODEL_CALIBRATION_METHOD"] = args.calibration_method
    if args.min_vol_z is not None:
        os.environ["AEB_SPECIALIST_MIN_VOL_Z"] = str(args.min_vol_z)
    if args.walk_forward:
        os.environ["AEB_WALK_FORWARD"] = "1"
    if args.apply_regime_filter_at_inference:
        os.environ["AEB_APPLY_REGIME_FILTER_AT_INFERENCE"] = "1"
    side_configs_path = out_dir / "side_configs.json"
    side_pair.save(out_dir)
    os.environ["AEB_SIDE_CONFIGS_PATH"] = str(side_configs_path)

    mtf = [t.strip() for t in args.multi_timeframe.split(",") if t.strip()]
    if "15m" in mtf:
        import pandas as pd

        from ae_brain.features.mtf import compute_15m_features_for_1h, load_15m_candles

        mtf_cache: dict[str, dict[str, list[float]]] = {}
        for sym in symbols:
            p1h = data_dir / f"{sym}_{args.interval}.csv"
            if not p1h.exists():
                continue
            c1h = pd.read_csv(p1h)
            if "ts" not in c1h.columns and "timestamp" in c1h.columns:
                c1h = c1h.rename(columns={"timestamp": "ts"})
            c15 = load_15m_candles(data_dir, sym)
            feats = compute_15m_features_for_1h(c1h, c15)
            mtf_cache[sym] = {k: v.tolist() for k, v in feats.items()}
        mtf_path = out_dir / "mtf_15m_cache.json"
        mtf_path.write_text(json.dumps(mtf_cache), encoding="utf-8")
        os.environ["AEB_MTF_15M_CACHE_PATH"] = str(mtf_path)
        os.environ["AEB_MULTI_TIMEFRAME"] = "15m"
    if args.min_net_reward is not None:
        os.environ["AEB_LABEL_MIN_NET_REWARD"] = str(args.min_net_reward)
    os.environ["AEB_LABEL_HORIZON"] = str(args.horizon)
    os.environ["AEB_SPECIALIST_MODEL_KIND"] = args.specialist_model_kind
    os.environ["AEB_RISK_ATR_TP_MULT"] = str(args.tp_mult)
    os.environ["AEB_RISK_ATR_SL_MULT"] = str(args.sl_mult)

    # Side-balance config: propagated to train_side_specialists via env vars.
    if args.balance_side_specialists == "true":
        os.environ["AEB_BALANCE_SIDE_SPECIALISTS"] = "true"
        # Only forward explicit weights (not the 'auto' default -> lets the trainer
        # compute the per-side imbalance ratio itself).
        if args.long_positive_weight.strip().lower() != "auto":
            os.environ["AEB_LONG_POSITIVE_WEIGHT"] = str(args.long_positive_weight)
        if args.short_positive_weight.strip().lower() != "auto":
            os.environ["AEB_SHORT_POSITIVE_WEIGHT"] = str(args.short_positive_weight)
    if args.balance_train_samples == "true":
        os.environ["AEB_BALANCE_TRAIN_SAMPLES"] = "true"
        if args.max_side_train_samples_per_class is not None:
            os.environ["AEB_MAX_SIDE_TRAIN_SAMPLES_PER_CLASS"] = str(args.max_side_train_samples_per_class)
    if args.allow_skip_sequence == "true":
        os.environ["AEB_ALLOW_SKIP_SEQUENCE"] = "true"
    if args.skip_sequence == "true":
        os.environ["AEB_SKIP_SEQUENCE"] = "true"
    if args.skip_rl == "true":
        os.environ["AEB_SKIP_RL"] = "true"

    from scripts import train_production as tp

    seq_epochs = 1 if args.quick else args.seq_epochs
    rl_steps = 10_000 if args.quick else args.rl_timesteps
    sample_n = args.sample_per_symbol
    if args.medium:
        seq_epochs = 2
        rl_steps = 3_000
        sample_n = sample_n or 15_000

    argv = [
        "--data-dir",
        str(data_dir),
        "--symbols",
        ",".join(symbols),
        "--interval",
        args.interval,
        "--seq-epochs",
        str(seq_epochs),
        "--rl-timesteps",
        str(rl_steps),
        "--meta-mode",
        args.meta_mode,
    ]
    if sample_n:
        argv.extend(["--sample-per-symbol", str(sample_n)])
    if args.allow_skip_sequence == "true":
        argv.append("--allow-skip-sequence")
    if args.skip_sequence == "true":
        argv.extend(["--skip-sequence", "true"])
    if args.skip_rl == "true":
        argv.extend(["--skip-rl", "true"])
    # Delegate to existing orchestrator (preserves layer training code).
    sys.argv = ["train_production.py", *argv]
    tp.main()

    # Label distribution report on first available symbol CSV.
    try:
        import numpy as np
        import pandas as pd
        from ae_brain.features.engineering import FeatureEngineer
        from ae_brain.training.labels import compute_labels_for_frame

        lab_parts = []
        for sym in symbols:
            p = data_dir / f"{sym}_{args.interval}.csv"
            if not p.exists():
                continue
            candles = pd.read_csv(p)
            if "ts" not in candles.columns and "timestamp" in candles.columns:
                candles = candles.rename(columns={"timestamp": "ts"})
            eng = FeatureEngineer(z_window=100)
            feats = eng.compute_frame(candles)
            labels, _ = compute_labels_for_frame(
                candles,
                feats["atr_14"].to_numpy(float),
                cfg=label_config,
            )
            if "ts" not in candles.columns:
                raise ValueError(f"{p} missing ts/timestamp column for label report")
            ts = pd.to_datetime(candles["ts"], utc=True)
            lab_parts.append(pd.DataFrame({"label": labels, "timestamp": ts, "symbol": sym}))
        if lab_parts:
            lab = pd.concat(lab_parts, ignore_index=True)
            run.decision_distribution = label_distribution_report(
                lab["label"].to_numpy(), lab["timestamp"], lab["symbol"].to_numpy()
            )["label_distribution_overall"]
    except Exception as exc:
        run.train_metrics["label_report_error"] = str(exc)

    run.artifacts_path = str(out_dir)
    summary_path = out_dir / "training_summary.json"
    if summary_path.exists():
        run.train_metrics = json.loads(summary_path.read_text())
    tracker.log_run(run)
    print(f"Training complete. Artifacts: {out_dir}")

    if not args.skip_evaluate:
        import subprocess

        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_candidate.py"),
            "--run-id",
            run.run_id,
            "--dataset",
            str(args.dataset or ROOT / "data" / "datasets" / "multi_asset.parquet"),
            "--symbols",
            ",".join(symbols),
            "--interval",
            args.interval,
            "--tp-mult",
            str(args.tp_mult),
            "--sl-mult",
            str(args.sl_mult),
            "--horizon",
            str(args.horizon),
        ]
        if args.apply_regime_filter_at_inference:
            eval_cmd.append("--apply-regime-filter-at-inference")
        print("Running post-train evaluation:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=str(ROOT), check=False)

    if args.short_search:
        import subprocess

        search_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "search_short_configs.py"),
            "--base-artifacts",
            str(out_dir),
            "--dataset",
            str(args.dataset or ROOT / "data" / "datasets" / "multi_asset.parquet"),
        ]
        if sample_n:
            search_cmd.extend(["--sample-per-symbol", str(sample_n)])
        print("Running SHORT config search:", " ".join(search_cmd))
        subprocess.run(search_cmd, cwd=str(ROOT), check=False)

    print(f"Evaluate: python scripts/evaluate_candidate.py --run-id {run.run_id}")
    print(f"Promote: python scripts/promote_model.py --run-id {run.run_id}")


if __name__ == "__main__":
    main()
