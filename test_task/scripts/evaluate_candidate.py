#!/usr/bin/env python3
"""Full evaluation for a trained candidate: backtest, diagnose, labels, meta audit."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.config import get_settings
from ae_brain.contracts import TradeCandidate
from ae_brain.inference.engine import InferenceEngine
from ae_brain.layers.meta import CLASS_LONG, CLASS_SHORT, CLASS_SKIP, MetaModel, build_meta_features, resolve_directional_class
from ae_brain.messaging.publish_gate import evaluate_publish
from ae_brain.messaging.skip_reason import extract_skip_reason
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.canonical import candles_from_canonical
from ae_brain.training.dataset import directional_barrier_labels, relative_vol_scale
from ae_brain.training.evaluation import (
    SignalBatch,
    build_evaluation_report,
    build_summary_json,
    build_test_metrics_payload,
    evaluate_meta_predictions,
    side_specialist_collapse_warnings,
    split_label_counts,
)
from ae_brain.training.labels import LabelConfig, compute_labels_for_frame, label_distribution_report
from ae_brain.training.regime_filter import TrainingRegimeConfig, load_training_regime
from ae_brain.training.side_configs import load_side_configs
from ae_brain.training.splits import make_time_split
from scripts.run_backtest import collect_backtest_signals


def _regime_filter_evaluation(
    batch,
    *,
    regime: TrainingRegimeConfig,
    vol_z_by_row: np.ndarray | None,
) -> dict:
    if vol_z_by_row is None or regime.min_vol_z is None:
        return {"applied": False}
    below = vol_z_by_row < regime.min_vol_z
    kept = ~below
    pub_before_l = int(((batch.decisions == "LONG") & batch.publishable).sum())
    pub_before_s = int(((batch.decisions == "SHORT") & batch.publishable).sum())
    ev_before_l = float(batch.expected_evs[(batch.decisions == "LONG") & batch.publishable].sum())
    ev_before_s = float(batch.expected_evs[(batch.decisions == "SHORT") & batch.publishable].sum())
    pub_after_l = int(((batch.decisions == "LONG") & batch.publishable & kept).sum())
    pub_after_s = int(((batch.decisions == "SHORT") & batch.publishable & kept).sum())
    ev_after_l = float(batch.expected_evs[(batch.decisions == "LONG") & batch.publishable & kept].sum())
    ev_after_s = float(batch.expected_evs[(batch.decisions == "SHORT") & batch.publishable & kept].sum())
    return {
        "applied": True,
        "min_vol_z": regime.min_vol_z,
        "test_bars_below_min_vol_z": int(below.sum()),
        "publishable_LONG_before_filter": pub_before_l,
        "publishable_SHORT_before_filter": pub_before_s,
        "publishable_LONG_after_filter": pub_after_l,
        "publishable_SHORT_after_filter": pub_after_s,
        "publishable_EV_LONG_before": ev_before_l,
        "publishable_EV_SHORT_before": ev_before_s,
        "publishable_EV_LONG_after": ev_after_l,
        "publishable_EV_SHORT_after": ev_after_s,
        "min_vol_z_mismatch_affects_test": int(below.sum()) > 0
        and (pub_before_l != pub_after_l or pub_before_s != pub_after_s),
    }


def _test_vol_z_series(dataset: Path, symbols: list[str], batch_timestamps: np.ndarray, batch_symbols: np.ndarray) -> np.ndarray:
    from ae_brain.features.engineering import FeatureEngineer

    df = pd.read_parquet(dataset)
    df = df[df["symbol"].isin(symbols)].sort_values(["symbol", "timestamp"])
    split = make_time_split(df["timestamp"])
    test_df = df.iloc[split.test_idx]
    vol_z_map: dict[tuple[str, str], float] = {}
    for sym in sorted(test_df["symbol"].unique()):
        sub = test_df[test_df["symbol"] == sym]
        candles = candles_from_canonical(sub)
        feats = FeatureEngineer(z_window=100).compute_frame(candles)
        ts = pd.to_datetime(candles["ts"], utc=True)
        for i in range(len(ts)):
            key = (sym, str(ts.iloc[i]))
            vol_z_map[key] = float(feats["vol_z"].iloc[i])
    out = []
    for sym, ts in zip(batch_symbols, batch_timestamps):
        out.append(vol_z_map.get((str(sym), str(ts)), 0.0))
    return np.asarray(out, dtype=float)


async def _diagnose_batch(
    dataset: Path,
    artifacts: Path,
    symbols: list[str],
    thresholds: list[float],
    window: int,
) -> dict:
    settings = get_settings()
    settings.model.artifacts_dir = artifacts
    engine = InferenceEngine(settings, db=None)
    engine.load_models()

    df = pd.read_parquet(dataset)
    split = make_time_split(df["timestamp"])
    test_df = df.iloc[split.test_idx]

    btc_sub = df[df["symbol"] == "BTCUSDT"].sort_values("timestamp")
    btc_ctx_by_ts: dict = {}
    if not btc_sub.empty:
        from ae_brain.features.engineering import FeatureEngineer
        from ae_brain.features.schema import REGIME_ONEHOT_NAMES

        btc_candles = candles_from_canonical(btc_sub)
        btc_eng = FeatureEngineer(z_window=100)
        btc_feats = btc_eng.compute_frame(btc_candles)
        btc_ts = pd.to_datetime(btc_candles["ts"], utc=True)
        trend_col = REGIME_ONEHOT_NAMES[0]
        for j in range(len(btc_ts)):
            btc_ctx_by_ts[btc_ts.iloc[j]] = {
                "btc_ret_15": float(btc_feats["ret_15"].iloc[j]),
                "btc_vol_z": float(btc_feats["vol_z"].iloc[j]),
                "btc_regime_trend": float(btc_feats[trend_col].iloc[j]) if trend_col in btc_feats.columns else 0.0,
            }

    internal = {"LONG": 0, "SHORT": 0, "SKIP": 0}
    published = {t: {"LONG": 0, "SHORT": 0} for t in thresholds}
    suppressed = {t: 0 for t in thresholds}
    skip_reasons: dict[str, int] = {}
    per_symbol: dict[str, dict[str, int]] = {}

    for sym in sorted(test_df["symbol"].unique()):
        if sym not in symbols:
            continue
        sub = test_df[test_df["symbol"] == sym]
        frame = candles_from_canonical(sub)
        if len(frame) < window + 50:
            continue
        per_symbol.setdefault(sym, {"LONG": 0, "SHORT": 0, "SKIP": 0})
        for i in range(window, len(frame), max(24, len(frame) // 20)):
            chunk = frame.iloc[i - window : i + 1].copy()
            ts_key = pd.to_datetime(chunk["ts"].iloc[-1], utc=True)
            meta = {
                "current_price": float(chunk["close"].iloc[-1]),
                "composite_score": 0.8,
                "features": {"current_price": float(chunk["close"].iloc[-1])},
            }
            if sym != "BTCUSDT" and btc_ctx_by_ts:
                meta["btc_specialist_ctx"] = btc_ctx_by_ts.get(ts_key, {})
            cand = TradeCandidate.from_message(
                {
                    "symbol": sym,
                    "interval": "1h",
                    "asset_class": "crypto",
                    "candles": chunk.assign(ts=chunk["ts"].astype(str)).to_dict(orient="records"),
                    "meta": meta,
                }
            )
            signal = await engine.evaluate(cand)
            d = signal.decision.value
            internal[d] = internal.get(d, 0) + 1
            per_symbol[sym][d] = per_symbol[sym].get(d, 0) + 1
            if d == "SKIP":
                reason = extract_skip_reason(signal) or "unknown"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            for t in thresholds:
                ok, _, _ = evaluate_publish(signal, allowed_symbols=frozenset(symbols), min_confidence=t)
                if ok and d in ("LONG", "SHORT"):
                    published[t][d] += 1
                elif d in ("LONG", "SHORT"):
                    suppressed[t] += 1

    await engine.shutdown()
    return {
        "internal_model_signals": internal,
        "telegram_publishable_signals": {f"confidence_ge_{t:.2f}": published[t] for t in thresholds},
        "suppressed_low_confidence_signals": {f"threshold_{t:.2f}": suppressed[t] for t in thresholds},
        "per_symbol_decision_distribution": per_symbol,
        "skip_reasons": skip_reasons,
        "artifacts": str(artifacts),
    }


def _training_label_audit(dataset: Path, symbols: list[str], interval: str, cfg: LabelConfig) -> dict:
    df = pd.read_parquet(dataset)
    df = df[(df["symbol"].isin(symbols)) & (df["timeframe"] == interval)].sort_values(["symbol", "timestamp"])
    parts = []
    for sym in symbols:
        sub = df[df["symbol"] == sym]
        if sub.empty:
            continue
        candles = candles_from_canonical(sub)
        close = candles["close"].to_numpy(float)
        atr = np.maximum(
            candles["high"].to_numpy(float) - candles["low"].to_numpy(float),
            close * 0.005,
        )
        labels, _ = compute_labels_for_frame(candles, atr, cfg=cfg)
        ts = pd.to_datetime(candles["ts"], utc=True)
        parts.append(pd.DataFrame({"label": labels, "timestamp": ts, "symbol": sym}))
    if not parts:
        return {}
    lab = pd.concat(parts, ignore_index=True)
    overall = label_distribution_report(lab["label"].to_numpy(), lab["timestamp"], lab["symbol"].to_numpy())
    overall["split_counts"] = split_label_counts(lab, label_col="label")
    return overall


def _meta_model_audit(artifacts: Path, dataset: Path, symbols: list[str]) -> dict:
    meta = MetaModel().load(artifacts)
    if not meta.is_ready():
        return {"error": "meta_model_not_ready"}

    df = pd.read_parquet(dataset)
    df = df[df["symbol"].isin(symbols)].sort_values(["symbol", "timestamp"])
    split = make_time_split(df["timestamp"])
    test_df = df.iloc[split.test_idx]

    settings = get_settings()
    y_true, y_pred, proba_rows = [], [], []
    threshold = settings.fusion.meta_direction_threshold

    for sym in sorted(test_df["symbol"].unique()):
        sub = test_df[test_df["symbol"] == sym]
        candles = candles_from_canonical(sub)
        atr = candles["close"].to_numpy(float) * 0.01
        if "high" in candles.columns:
            atr = np.maximum(
                candles["high"].to_numpy(float) - candles["low"].to_numpy(float),
                candles["close"].to_numpy(float) * 0.005,
            )
        vol_scale = relative_vol_scale(atr / candles["close"].to_numpy(float))
        labels = directional_barrier_labels(
            candles,
            atr,
            tp_mult=settings.risk.atr_tp_mult,
            sl_mult=settings.risk.atr_sl_mult,
            horizon=24,
            vol_scale=vol_scale,
        )
        # Use neutral base-layer inputs to isolate meta bias from base models.
        for i in range(100, len(candles) - 24, 48):
            reg = np.array([0.33, 0.34, 0.33], dtype=float)
            vec = build_meta_features(0.5, 0.5, 1.0, 0.0, reg)
            pred = meta.predict(vec)
            proba = np.array([pred.p_short, pred.p_skip, pred.p_long], dtype=float)
            directional, _ = resolve_directional_class(pred.p_short, pred.p_long, threshold=threshold)
            y_true.append(int(labels[i]))
            y_pred.append(int(directional if directional is not None else CLASS_SKIP))
            proba_rows.append(proba)

    if not y_true:
        return {"error": "no_meta_audit_rows"}
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    proba_arr = np.asarray(proba_rows, dtype=float)
    audit = evaluate_meta_predictions(y_true_arr, y_pred_arr, proba_arr)
    audit["meta_direction_threshold"] = threshold
    audit["note"] = (
        "Audit uses neutral base-layer features; see backtest meta components for live p_long/p_short."
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate model run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates-dir", type=Path, default=ROOT / "artifacts_candidates")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--publish-confidence", type=float, default=0.70)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--tp-mult", type=float, default=2.5)
    parser.add_argument("--sl-mult", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--apply-regime-filter-at-inference", action="store_true")
    args = parser.parse_args()

    artifacts = args.candidates_dir / args.run_id
    if not artifacts.is_dir():
        print(f"Candidate not found: {artifacts}", file=sys.stderr)
        sys.exit(1)
    if not args.dataset.exists():
        print(f"Dataset missing: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    side_configs = load_side_configs(artifacts)
    label_cfg = side_configs.long.to_label_config()
    regime = load_training_regime(artifacts)
    if args.apply_regime_filter_at_inference:
        regime = TrainingRegimeConfig(min_vol_z=regime.min_vol_z, apply_at_inference=True)
        (artifacts / "training_regime.json").write_text(json.dumps(regime.to_dict(), indent=2), encoding="utf-8")

    label_report = _training_label_audit(args.dataset, symbols, args.interval, label_cfg)
    meta_eval = _meta_model_audit(artifacts, args.dataset, symbols)

    training_summary_path = artifacts / "training_summary.json"
    meta_mode = "two_stage"
    if training_summary_path.exists():
        meta_mode = json.loads(training_summary_path.read_text(encoding="utf-8")).get("meta", {}).get(
            "meta_mode", meta_mode
        )

    collect_kwargs: dict = {}
    if meta_mode == "side_aware_ensemble":
        collect_kwargs["side_aware_ensemble"] = True
    elif meta_mode == "side_specialists":
        collect_kwargs["side_specialists"] = True

    batch, _ = asyncio.run(
        collect_backtest_signals(args.dataset, artifacts, symbols, args.publish_confidence, **collect_kwargs)
    )
    vol_z_rows = _test_vol_z_series(args.dataset, symbols, batch.timestamps, batch.symbols)
    backtest_report = build_evaluation_report(
        batch,
        publish_confidence=args.publish_confidence,
        label_report=label_report,
        meta_eval=meta_eval,
    )
    backtest_report["regime_filter_evaluation"] = _regime_filter_evaluation(
        batch, regime=regime, vol_z_by_row=vol_z_rows
    )
    backtest_report["side_configs"] = side_configs.to_dict()

    # Meta probability stats from live backtest batch.
    if batch.meta_p_long is not None:
        actionable = np.isin(batch.decisions, ["LONG", "SHORT"])
        backtest_report["live_meta_probability_stats"] = {
            "p_long_actionable": {
                "mean": float(np.nanmean(batch.meta_p_long[actionable])) if actionable.any() else None,
                "max": float(np.nanmax(batch.meta_p_long[actionable])) if actionable.any() else None,
            },
            "p_short_actionable": {
                "mean": float(np.nanmean(batch.meta_p_short[actionable])) if actionable.any() else None,
                "max": float(np.nanmax(batch.meta_p_short[actionable])) if actionable.any() else None,
            },
            "both_directional_pass_rate": float(
                np.nanmean((batch.meta_p_long > 0.30) & (batch.meta_p_short > 0.30))
            ),
            "long_wins_tiebreak_rate": float(
                np.nanmean(
                    (batch.meta_p_long > 0.30)
                    & (batch.meta_p_short > 0.30)
                    & (batch.meta_p_long >= batch.meta_p_short)
                )
            ),
        }

    diagnose_report = asyncio.run(
        _diagnose_batch(args.dataset, artifacts, symbols, [0.50, 0.60, 0.70, 0.80], window=48)
    )
    diagnose_report["promotable"] = backtest_report.get("promotable", False)
    diagnose_report["promotion_blockers"] = backtest_report.get("promotion_blockers", [])

    training_metrics = {}
    training_log = args.report_dir / "train_multi_asset_1h_full_lite.log"
    run_json = args.report_dir / f"run_{args.run_id}.json"
    if run_json.exists():
        training_metrics["experiment_run"] = json.loads(run_json.read_text())
    if training_log.exists():
        training_metrics["training_log"] = str(training_log)

    summary = build_summary_json(
        args.run_id,
        backtest_report,
        artifacts_path=str(artifacts),
        training_metrics=training_metrics,
    )
    summary["success"] = summary["promotable"]
    if meta_mode == "side_specialists":
        collapse_warnings = side_specialist_collapse_warnings(
            artifacts,
            summary,
            publish_confidence=args.publish_confidence,
        )
        if collapse_warnings:
            summary.setdefault("warnings", [])
            for w in collapse_warnings:
                if w not in summary["warnings"]:
                    summary["warnings"].append(w)
        summary["decision_mode"] = "side_specialists_calibrated_prob_direct"

    args.report_dir.mkdir(parents=True, exist_ok=True)
    test_metrics_payload = build_test_metrics_payload(backtest_report)
    (artifacts / "test_metrics.json").write_text(
        json.dumps(test_metrics_payload, indent=2),
        encoding="utf-8",
    )
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.report_dir / f"backtest_{args.run_id}.json").write_text(json.dumps(backtest_report, indent=2), encoding="utf-8")
    (args.report_dir / f"diagnose_{args.run_id}.json").write_text(json.dumps(diagnose_report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not summary["promotable"]:
        print("NOT PROMOTABLE:", summary["promotion_blockers"], file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
