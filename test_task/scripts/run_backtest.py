#!/usr/bin/env python3
"""Walk-forward backtest evaluator with EV-first metrics and publish-gate separation."""

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
from ae_brain.contracts import Decision, TradeCandidate
from ae_brain.inference.engine import InferenceEngine
from ae_brain.messaging.publish_gate import evaluate_publish
from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.execution.timing import ms_to_iso_utc
from ae_brain.training.canonical import candles_from_canonical
from ae_brain.training.evaluation import SignalBatch, build_evaluation_report
from ae_brain.training.splits import make_time_split


async def collect_backtest_signals(
    dataset: Path,
    artifacts: Path,
    symbols: list[str],
    publish_conf: float,
    *,
    step: int = 24,
    ablation_mode: str | None = None,
    use_legacy_meta: bool = False,
    no_meta: bool = False,
    side_aware_ensemble: bool = False,
    side_specialists: bool = False,
) -> tuple[SignalBatch, pd.DataFrame]:
    df = pd.read_parquet(dataset)
    df = df[df["symbol"].isin(symbols)].sort_values(["symbol", "timestamp"])
    split = make_time_split(df["timestamp"])
    test_df = df.iloc[split.test_idx].copy()

    settings = get_settings()
    settings.model.artifacts_dir = artifacts
    engine = InferenceEngine(settings, db=None)
    engine.load_models()
    if no_meta:
        engine._fusion._meta = None
        engine._fusion._ablation_mode = ablation_mode
    elif use_legacy_meta:
        from ae_brain.layers.meta import MetaModel

        engine._fusion._meta = MetaModel().load(artifacts)
        engine._fusion._ablation_mode = ablation_mode
    elif side_specialists:
        from ae_brain.layers.side_specialists import load_side_specialists

        settings.fusion.meta_mode = "side_specialists"
        engine._fusion._force_meta_mode = "side_specialists"
        engine._fusion._side_specialists = load_side_specialists(artifacts)
        engine._fusion._side_calibrators.load(artifacts)
    elif side_aware_ensemble:
        from ae_brain.layers.meta import MetaModel, TwoStageMetaModel
        from ae_brain.layers.side_aware import load_side_aware_config

        settings.fusion.meta_mode = "side_aware_ensemble"
        engine._fusion._force_meta_mode = "side_aware_ensemble"
        engine._fusion._side_aware_config = load_side_aware_config(artifacts)
        engine._fusion._side_calibrators.load(artifacts)
        engine._fusion._meta_legacy = MetaModel().load(artifacts)
        engine._fusion._meta_two_stage = TwoStageMetaModel().load(artifacts)
        if not engine._fusion._meta_legacy.is_ready():
            engine._fusion._meta_legacy = None
        if not engine._fusion._meta_two_stage.is_ready():
            engine._fusion._meta_two_stage = None
    elif ablation_mode:
        engine._fusion._ablation_mode = ablation_mode
        if ablation_mode in ("tabular_only", "tabular_sequence", "tabular_rl", "full_no_meta"):
            engine._fusion._meta = None

    decisions, evs, confs, syms, times, publishable = [], [], [], [], [], []
    meta_p_short, meta_p_long, meta_p_skip = [], [], []
    fused_scores, tabular_p_up, raw_long_conf, raw_short_conf = [], [], [], []
    window = settings.model.sequence_window

    for sym in sorted(test_df["symbol"].unique()):
        sub = test_df[test_df["symbol"] == sym]
        candles = candles_from_canonical(sub)
        for i in range(window, len(candles) - 1, step):
            chunk = candles.iloc[i - window : i + 1]
            next_row = candles.iloc[i + 1]
            rows = chunk.assign(ts=chunk["ts"].astype(str)).to_dict(orient="records")
            signal_close = float(chunk["close"].iloc[-1])
            signal_open_ts = chunk["ts"].iloc[-1]
            next_open = float(next_row["open"])
            next_open_ts = next_row["ts"]
            cand = TradeCandidate.from_message(
                {
                    "symbol": sym,
                    "interval": "1h",
                    "asset_class": "crypto",
                    "candles": rows,
                    "meta": {
                        "current_price": signal_close,
                        "composite_score": 0.8,
                        "features": {"current_price": signal_close},
                        "signal_candle_open_time": (
                            signal_open_ts.isoformat() if hasattr(signal_open_ts, "isoformat") else str(signal_open_ts)
                        ),
                        "signal_candle_close_time": ms_to_iso_utc(
                            int(signal_open_ts.timestamp() * 1000) + 3_599_999
                        )
                        if hasattr(signal_open_ts, "timestamp")
                        else str(signal_open_ts),
                        "next_candle_open": next_open,
                        "next_candle_open_time": (
                            next_open_ts.isoformat() if hasattr(next_open_ts, "isoformat") else str(next_open_ts)
                        ),
                    },
                }
            )
            sig = await engine.evaluate(cand)
            decisions.append(sig.decision.value)
            evs.append(sig.expected_value_usd)
            confs.append(sig.confidence)
            syms.append(sym)
            ts_val = chunk["ts"].iloc[-1]
            times.append(ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val))
            ok, _, _ = evaluate_publish(sig, allowed_symbols=frozenset(symbols), min_confidence=publish_conf)
            publishable.append(ok)
            comp = sig.components or {}
            meta = comp.get("meta") or {}
            meta_p_short.append(float(meta.get("p_short_given_trade", meta.get("p_short", np.nan))))
            meta_p_long.append(float(meta.get("p_long_given_trade", meta.get("p_long", np.nan))))
            meta_p_skip.append(float(meta.get("p_trade", meta.get("p_skip", np.nan))))
            sa = comp.get("side_aware") or {}
            ss = comp.get("side_specialists") or {}
            lc = sa.get("long_candidate") or ss.get("long") or {}
            sc = sa.get("short_candidate") or ss.get("short") or {}
            if sa:
                fused_scores.append(float(lc.get("fused_score", sc.get("fused_score", np.nan))))
                raw_long_conf.append(float(lc.get("raw_confidence", lc.get("p_long_profitable_raw", np.nan))))
                raw_short_conf.append(float(sc.get("raw_confidence", sc.get("p_short_profitable_raw", np.nan))))
            elif ss:
                fused_scores.append(np.nan)
                raw_long_conf.append(float(lc.get("p_long_profitable_raw", np.nan)))
                raw_short_conf.append(float(sc.get("p_short_profitable_raw", np.nan)))
            else:
                fused_scores.append(float(comp.get("fused_score", np.nan)))
                raw_long_conf.append(float(meta.get("p_long_given_trade", meta.get("p_long", np.nan))))
                raw_short_conf.append(float(meta.get("p_short_given_trade", meta.get("p_short", np.nan))))
            lp = comp.get("layer_probs") or {}
            tabular_p_up.append(float(lp.get("tabular_p_up", np.nan)))

    await engine.shutdown()
    batch = SignalBatch(
        decisions=np.array(decisions),
        expected_evs=np.array(evs, dtype=float),
        confidence=np.array(confs, dtype=float),
        symbols=np.array(syms),
        timestamps=np.array(times),
        publishable=np.array(publishable, dtype=bool),
        meta_p_short=np.array(meta_p_short, dtype=float),
        meta_p_long=np.array(meta_p_long, dtype=float),
        meta_p_skip=np.array(meta_p_skip, dtype=float),
        fused_scores=np.array(fused_scores, dtype=float),
        tabular_p_up=np.array(tabular_p_up, dtype=float),
        raw_long_confidence=np.array(raw_long_conf, dtype=float),
        raw_short_confidence=np.array(raw_short_conf, dtype=float),
    )
    return batch, test_df


async def run_backtest(
    dataset: Path,
    artifacts: Path,
    symbols: list[str],
    publish_conf: float,
    *,
    label_report: dict | None = None,
    meta_eval: dict | None = None,
) -> dict:
    batch, _ = await collect_backtest_signals(dataset, artifacts, symbols, publish_conf)
    return build_evaluation_report(
        batch,
        publish_confidence=publish_conf,
        label_report=label_report,
        meta_eval=meta_eval,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--publish-confidence", type=float, default=0.70)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--write-test-metrics", type=Path, default=None, help="Also write test_metrics.json to this dir")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Missing dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    report = asyncio.run(run_backtest(args.dataset, args.artifacts, symbols, args.publish_confidence))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.run_id}" if args.run_id else ""
    out_path = args.report_dir / f"backtest{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.write_test_metrics is not None:
        args.write_test_metrics.mkdir(parents=True, exist_ok=True)
        internal = report.get("backtest_internal_all_signals") or {}
        (args.write_test_metrics / "test_metrics.json").write_text(json.dumps(internal, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
