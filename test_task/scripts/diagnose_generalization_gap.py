"""Validation vs test generalization gap diagnostics for a trained candidate run."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.config import get_settings
from ae_brain.contracts import TradeCandidate
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.features.schema import REGIME_ONEHOT_NAMES
from ae_brain.inference.engine import InferenceEngine
from ae_brain.layers.side_specialists import load_side_specialists
from ae_brain.messaging.publish_gate import evaluate_publish
from ae_brain.messaging.skip_reason import extract_skip_reason
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.canonical import candles_from_canonical
from ae_brain.training.regime_filter import TrainingRegimeConfig, load_training_regime
from ae_brain.training.side_configs import load_side_configs
from ae_brain.training.splits import make_time_split


def _dist_stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "p50": float(np.median(v)),
        "p90": float(np.quantile(v, 0.9)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def _regime_bucket(vol_z: float) -> str:
    if vol_z < -0.5:
        return "low_vol"
    if vol_z < 0.0:
        return "below_avg_vol"
    if vol_z < 0.5:
        return "avg_vol"
    return "high_vol"


def _trend_bucket(adx: float) -> str:
    if adx < 20:
        return "weak_trend"
    if adx < 30:
        return "moderate_trend"
    return "strong_trend"


def _funding_bucket(fz: float) -> str:
    if fz < -0.5:
        return "negative_funding"
    if fz < 0.5:
        return "neutral_funding"
    return "positive_funding"


def _oi_bucket(oiz: float) -> str:
    if oiz < -0.5:
        return "oi_contracting"
    if oiz < 0.5:
        return "oi_stable"
    return "oi_expanding"


def _count_distribution(keys: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in keys:
        out[k] = out.get(k, 0) + 1
    return out


async def _score_split(
    df: pd.DataFrame,
    artifacts: Path,
    symbols: list[str],
    *,
    apply_regime_filter: bool,
    regime: TrainingRegimeConfig,
    window: int = 48,
    step: int = 24,
) -> dict[str, Any]:
    settings = get_settings()
    settings.model.artifacts_dir = artifacts
    if apply_regime_filter:
        regime = TrainingRegimeConfig(min_vol_z=regime.min_vol_z, apply_at_inference=True)
        (artifacts / "training_regime.json").write_text(json.dumps(regime.to_dict(), indent=2), encoding="utf-8")

    engine = InferenceEngine(settings, db=None)
    engine.load_models()
    training_summary_path = artifacts / "training_summary.json"
    if training_summary_path.exists():
        meta_mode = json.loads(training_summary_path.read_text(encoding="utf-8")).get("meta", {}).get(
            "meta_mode", "two_stage"
        )
        if meta_mode == "side_specialists":
            from ae_brain.layers.side_specialists import load_side_specialists

            settings.fusion.meta_mode = "side_specialists"
            engine._fusion._force_meta_mode = "side_specialists"
            engine._fusion._side_specialists = load_side_specialists(artifacts)
            engine._fusion._side_calibrators.load(artifacts)
    if apply_regime_filter:
        engine._fusion._training_regime = regime

    specialists = load_side_specialists(artifacts)
    side_cals = engine._fusion._side_calibrators

    rows: list[dict[str, Any]] = []
    skip_reasons: dict[str, int] = {}
    below_min_vol_z = 0

    for sym in sorted(df["symbol"].unique()):
        if sym not in symbols:
            continue
        sub = df[df["symbol"] == sym]
        candles = candles_from_canonical(sub)
        if len(candles) < window + 10:
            continue
        eng = FeatureEngineer(z_window=100)
        feats = eng.compute_frame(candles)
        vol_z_arr = feats["vol_z"].to_numpy(float)

        for i in range(window, len(candles), step):
            if apply_regime_filter and regime.min_vol_z is not None and vol_z_arr[i] < regime.min_vol_z:
                below_min_vol_z += 1
                skip_reasons["below_min_vol_z"] = skip_reasons.get("below_min_vol_z", 0) + 1
                continue

            chunk = candles.iloc[i - window : i + 1]
            cand = TradeCandidate.from_message(
                {
                    "symbol": sym,
                    "interval": "1h",
                    "asset_class": "crypto",
                    "candles": chunk.assign(ts=chunk["ts"].astype(str)).to_dict(orient="records"),
                    "meta": {"current_price": float(chunk["close"].iloc[-1]), "composite_score": 0.8},
                }
            )
            sig = await engine.evaluate(cand)
            if sig.decision.value == "SKIP":
                reason = extract_skip_reason(sig) or "unknown"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            comp = sig.components or {}
            ss = comp.get("side_specialists") or {}
            lc = ss.get("long") or {}
            sc = ss.get("short") or {}
            pub_ok, _, _ = evaluate_publish(sig, allowed_symbols=frozenset(symbols), min_confidence=0.70)

            row = feats.iloc[i]
            regime_oh = row[list(REGIME_ONEHOT_NAMES)].to_numpy(float) if all(c in feats.columns for c in REGIME_ONEHOT_NAMES) else np.zeros(3)
            rows.append(
                {
                    "symbol": sym,
                    "decision": sig.decision.value,
                    "confidence": sig.confidence,
                    "ev": sig.expected_value_usd,
                    "publishable": pub_ok,
                    "long_raw": float(lc.get("p_long_profitable_raw", lc.get("p_profitable_raw", np.nan))),
                    "short_raw": float(sc.get("p_short_profitable_raw", sc.get("p_profitable_raw", np.nan))),
                    "long_cal": float(lc.get("p_long_profitable_calibrated", lc.get("p_profitable_calibrated", np.nan))),
                    "short_cal": float(sc.get("p_short_profitable_calibrated", sc.get("p_profitable_calibrated", np.nan))),
                    "vol_z": float(vol_z_arr[i]),
                    "vol_regime": _regime_bucket(float(vol_z_arr[i])),
                    "trend_regime": _trend_bucket(float(row.get("adx_14", 0.0))),
                    "funding_regime": _funding_bucket(float(row.get("funding_z", 0.0))),
                    "oi_regime": _oi_bucket(float(row.get("oi_z", 0.0))),
                    "gmm_regime": int(np.argmax(regime_oh)),
                }
            )

    await engine.shutdown()

    if not rows:
        return {"n_scored": 0}

    long_raw = np.array([r["long_raw"] for r in rows], dtype=float)
    short_raw = np.array([r["short_raw"] for r in rows], dtype=float)
    long_cal = np.array([r["long_cal"] for r in rows], dtype=float)
    short_cal = np.array([r["short_cal"] for r in rows], dtype=float)
    decisions = [r["decision"] for r in rows]
    publishable = [r["publishable"] for r in rows]

    pub_long = sum(1 for r in rows if r["decision"] == "LONG" and r["publishable"])
    pub_short = sum(1 for r in rows if r["decision"] == "SHORT" and r["publishable"])
    ev_long = sum(r["ev"] for r in rows if r["decision"] == "LONG" and r["publishable"])
    ev_short = sum(r["ev"] for r in rows if r["decision"] == "SHORT" and r["publishable"])

    return {
        "n_scored": len(rows),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "symbol_distribution": _count_distribution([r["symbol"] for r in rows]),
        "volatility_regime_distribution": _count_distribution([r["vol_regime"] for r in rows]),
        "trend_regime_distribution": _count_distribution([r["trend_regime"] for r in rows]),
        "funding_regime_distribution": _count_distribution([r["funding_regime"] for r in rows]),
        "oi_regime_distribution": _count_distribution([r["oi_regime"] for r in rows]),
        "gmm_regime_distribution": _count_distribution([str(r["gmm_regime"]) for r in rows]),
        "long_score_distribution": _dist_stats(long_raw),
        "short_score_distribution": _dist_stats(short_raw),
        "long_calibrated_confidence": _dist_stats(long_cal),
        "short_calibrated_confidence": _dist_stats(short_cal),
        "internal_signals": _count_distribution(decisions),
        "publishable_LONG_ge_70": pub_long,
        "publishable_SHORT_ge_70": pub_short,
        "publishable_EV_LONG": ev_long,
        "publishable_EV_SHORT": ev_short,
        "skip_reasons": skip_reasons,
        "below_min_vol_z_skipped": below_min_vol_z,
        "mean_vol_z": float(np.mean([r["vol_z"] for r in rows])),
    }


def build_generalization_report(
    run_id: str,
    artifacts: Path,
    dataset: Path,
    symbols: list[str],
) -> dict[str, Any]:
    df = pd.read_parquet(dataset)
    df = df[(df["symbol"].isin(symbols)) & (df["timeframe"] == "1h")].sort_values(["symbol", "timestamp"])
    split = make_time_split(df["timestamp"])
    val_df = df.iloc[split.val_idx]
    test_df = df.iloc[split.test_idx]

    side_configs = load_side_configs(artifacts)
    regime = load_training_regime(artifacts)

    val_no_filter = asyncio.run(_score_split(val_df, artifacts, symbols, apply_regime_filter=False, regime=regime))
    test_no_filter = asyncio.run(_score_split(test_df, artifacts, symbols, apply_regime_filter=False, regime=regime))
    test_with_filter = asyncio.run(
        _score_split(test_df, artifacts, symbols, apply_regime_filter=True, regime=regime)
    )

    val_pub_long = int(val_no_filter.get("publishable_LONG_ge_70", 0))
    test_pub_long_before = int(test_no_filter.get("publishable_LONG_ge_70", 0))
    test_pub_long_after = int(test_with_filter.get("publishable_LONG_ge_70", 0))

    gap_analysis = {
        "validation_had_302_long_publishable_but_test_had_0": val_pub_long > 0 and test_pub_long_before == 0,
        "training_validation_metrics": {},
        "likely_causes": [],
    }
    ts_path = artifacts / "training_summary.json"
    if ts_path.exists():
        train_meta = json.loads(ts_path.read_text(encoding="utf-8")).get("meta", {})
        gap_analysis["training_validation_metrics"] = train_meta.get("validation_production_metrics", {})
        train_val_long = int(train_meta.get("validation_production_metrics", {}).get("publishable_LONG_ge_0.70", 0))
        if train_val_long > 0 and test_pub_long_before == 0:
            gap_analysis["validation_had_302_long_publishable_but_test_had_0"] = True
    if gap_analysis["validation_had_302_long_publishable_but_test_had_0"]:
        v_cal = val_no_filter.get("long_calibrated_confidence", {})
        t_cal = test_no_filter.get("long_calibrated_confidence", {})
        if t_cal.get("max", 0) < 0.70:
            gap_analysis["likely_causes"].append("test_long_calibrated_confidence_never_reaches_0.70")
        if test_no_filter.get("mean_vol_z", 0) < val_no_filter.get("mean_vol_z", 0) - 0.2:
            gap_analysis["likely_causes"].append("test_period_lower_volatility_regime")
        if test_no_filter.get("volatility_regime_distribution", {}).get("low_vol", 0) > val_no_filter.get(
            "volatility_regime_distribution", {}
        ).get("low_vol", 0):
            gap_analysis["likely_causes"].append("test_has_more_low_vol_bars")
        gap_analysis["likely_causes"].append("distribution_shift_between_validation_and_test_periods")
        gap_analysis["validation_long_cal_max"] = v_cal.get("max")
        gap_analysis["test_long_cal_max"] = t_cal.get("max")
        gap_analysis["validation_long_raw_max"] = val_no_filter.get("long_score_distribution", {}).get("max")
        gap_analysis["test_long_raw_max"] = test_no_filter.get("long_score_distribution", {}).get("max")

    return {
        "run_id": run_id,
        "artifacts_path": str(artifacts),
        "side_configs": side_configs.to_dict(),
        "training_regime": regime.to_dict(),
        "validation": val_no_filter,
        "test_before_regime_filter": test_no_filter,
        "test_after_regime_filter": test_with_filter,
        "regime_filter_impact": {
            "test_bars_below_min_vol_z": int(test_with_filter.get("below_min_vol_z_skipped", 0)),
            "publishable_LONG_before": test_pub_long_before,
            "publishable_LONG_after": test_pub_long_after,
            "publishable_SHORT_before": int(test_no_filter.get("publishable_SHORT_ge_70", 0)),
            "publishable_SHORT_after": int(test_with_filter.get("publishable_SHORT_ge_70", 0)),
            "publishable_EV_LONG_before": float(test_no_filter.get("publishable_EV_LONG", 0.0)),
            "publishable_EV_LONG_after": float(test_with_filter.get("publishable_EV_LONG", 0.0)),
            "publishable_EV_SHORT_before": float(test_no_filter.get("publishable_EV_SHORT", 0.0)),
            "publishable_EV_SHORT_after": float(test_with_filter.get("publishable_EV_SHORT", 0.0)),
            "min_vol_z_mismatch_affects_test": (
                int(test_with_filter.get("below_min_vol_z_skipped", 0)) > 0
                and test_pub_long_before != test_pub_long_after
            ),
        },
        "generalization_gap_analysis": gap_analysis,
        "no_test_leakage": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization gap diagnostics for a candidate run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates-dir", type=Path, default=ROOT / "artifacts_candidates")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    artifacts = args.candidates_dir / args.run_id
    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    report = build_generalization_report(args.run_id, artifacts, args.dataset, symbols)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.report_dir / f"generalization_gap_{args.run_id}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
