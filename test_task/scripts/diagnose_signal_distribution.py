#!/usr/bin/env python3
"""Diagnose LONG/SHORT/SKIP imbalance and publish-gate effects."""

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
from ae_brain.messaging.publish_gate import evaluate_publish
from ae_brain.messaging.skip_reason import extract_skip_reason
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.canonical import candles_from_canonical
from ae_brain.training.synthetic import generate_synthetic_candles


def _candidate_from_candles(symbol: str, candles: pd.DataFrame) -> TradeCandidate:
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")
    price = float(candles["close"].iloc[-1])
    return TradeCandidate.from_message(
        {
            "symbol": symbol,
            "interval": "1h",
            "asset_class": "crypto",
            "candles": rows,
            "meta": {
                "current_price": price,
                "composite_score": 0.8,
                "features": {"current_price": price, "rsi": 50.0, "atr": float(candles["close"].diff().abs().mean() or 1)},
            },
        }
    )


async def _run_diagnosis(
    dataset: Path | None,
    artifacts: Path,
    symbols: list[str],
    thresholds: list[float],
    window: int,
) -> dict:
    settings = get_settings()
    settings.model.artifacts_dir = artifacts
    engine = InferenceEngine(settings, db=None)
    engine.load_models()

    internal = {"LONG": 0, "SHORT": 0, "SKIP": 0}
    published = {t: {"LONG": 0, "SHORT": 0} for t in thresholds}
    suppressed = {t: 0 for t in thresholds}
    skip_reasons: dict[str, int] = {}
    ev_by_decision: dict[str, list[float]] = {"LONG": [], "SHORT": [], "SKIP": []}
    conf_by_decision: dict[str, list[float]] = {"LONG": [], "SHORT": [], "SKIP": []}
    per_symbol: dict[str, dict[str, int]] = {}

    if dataset and dataset.exists():
        df = pd.read_parquet(dataset)
        groups = df.groupby("symbol")
    else:
        groups = ((s, generate_synthetic_candles(n=600, seed=hash(s) % 10000)) for s in symbols)

    for sym, frame in groups:
        if sym not in symbols:
            continue
        if isinstance(frame, pd.DataFrame) and "close" not in frame.columns:
            frame = candles_from_canonical(frame)
        elif not isinstance(frame, pd.DataFrame):
            frame = frame
        if len(frame) < window + 50:
            continue
        per_symbol.setdefault(sym, {"LONG": 0, "SHORT": 0, "SKIP": 0})
        for i in range(window, len(frame), max(24, len(frame) // 20)):
            chunk = frame.iloc[i - window : i + 1].copy()
            if "ts" not in chunk.columns:
                chunk["ts"] = chunk.get("timestamp", pd.RangeIndex(len(chunk)))
            cand = _candidate_from_candles(sym, chunk)
            signal = await engine.evaluate(cand)
            d = signal.decision.value
            internal[d] = internal.get(d, 0) + 1
            per_symbol[sym][d] = per_symbol[sym].get(d, 0) + 1
            ev_by_decision[d].append(float(signal.expected_value_usd))
            conf_by_decision[d].append(float(signal.confidence))
            if d == "SKIP":
                reason = extract_skip_reason(signal) or "unknown"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            for t in thresholds:
                ok, reason, _ = evaluate_publish(
                    signal,
                    allowed_symbols=frozenset(symbols),
                    min_confidence=t,
                )
                if ok and d in ("LONG", "SHORT"):
                    published[t][d] += 1
                elif d in ("LONG", "SHORT"):
                    suppressed[t] += 1

    await engine.shutdown()
    return {
        "internal_model_signals": internal,
        "telegram_publishable_signals": {
            f"confidence_ge_{t:.2f}": published[t] for t in thresholds
        },
        "suppressed_low_confidence_signals": {f"threshold_{t:.2f}": suppressed[t] for t in thresholds},
        "ev_distribution_by_decision": {k: {"mean": float(np.mean(v)), "n": len(v)} if v else {} for k, v in ev_by_decision.items()},
        "confidence_distribution_by_decision": {
            k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "n": len(v)} if v else {} for k, v in conf_by_decision.items()
        },
        "per_symbol_decision_distribution": per_symbol,
        "skip_reasons": skip_reasons,
        "artifacts": str(artifacts),
        "note": "Use threshold 0.70 for production Telegram/signal.final gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--confidence-thresholds", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    thresholds = [float(x) for x in args.confidence_thresholds.split(",") if x.strip()]
    dataset = args.dataset if args.dataset.exists() else None
    if dataset is None:
        print(f"Dataset missing ({args.dataset}); using synthetic candles per symbol.", file=sys.stderr)

    report = asyncio.run(_run_diagnosis(dataset, args.artifacts, symbols, thresholds, args.window))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "diagnose_signal_distribution.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
