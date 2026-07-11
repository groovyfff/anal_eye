#!/usr/bin/env python3
"""Ablation study: which stack component causes SHORT-only / low-confidence behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.evaluation import build_evaluation_report, build_mode_investigation, build_side_diagnostics
from scripts.run_backtest import collect_backtest_signals

ABLATIONS: dict[str, dict] = {
    "tabular_only": {"ablation_mode": "tabular_only", "no_meta": True},
    "no_meta": {"ablation_mode": None, "no_meta": True},
    "tabular_sequence": {"ablation_mode": "tabular_sequence", "no_meta": True},
    "tabular_rl": {"ablation_mode": "tabular_rl", "no_meta": True},
    "full_no_meta": {"ablation_mode": None, "no_meta": True},
    "legacy_3class_meta": {"ablation_mode": None, "use_legacy_meta": True},
    "two_stage_meta": {"ablation_mode": None, "no_meta": False},
    "full_two_stage": {"ablation_mode": None, "no_meta": False},
    "side_aware_ensemble": {"side_aware_ensemble": True},
}


async def _run_ablation(mode: str, dataset: Path, artifacts: Path, symbols: list[str], publish_conf: float) -> dict:
    cfg = ABLATIONS[mode]
    batch, _ = await collect_backtest_signals(
        dataset,
        artifacts,
        symbols,
        publish_conf,
        ablation_mode=cfg.get("ablation_mode"),
        use_legacy_meta=cfg.get("use_legacy_meta", False),
        no_meta=cfg.get("no_meta", False),
        side_aware_ensemble=cfg.get("side_aware_ensemble", False),
    )
    report = build_evaluation_report(batch, publish_confidence=publish_conf)
    report["ablation_mode"] = mode
    report["side_diagnostics"] = build_side_diagnostics(batch, publish_confidence=publish_conf)
    report["mode_investigation"] = build_mode_investigation(batch, publish_confidence=publish_conf)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--publish-confidence", type=float, default=0.70)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    results = {}
    for mode in ABLATIONS:
        print(f"Running ablation: {mode}", file=sys.stderr)
        try:
            results[mode] = asyncio.run(_run_ablation(mode, args.dataset, args.artifacts, symbols, args.publish_confidence))
        except Exception as exc:
            results[mode] = {"error": str(exc)}

    summary_rows = []
    for mode, rep in results.items():
        if "error" in rep:
            summary_rows.append({"mode": mode, "error": rep["error"]})
            continue
        internal = rep.get("internal_model_signals", {})
        pub = rep.get("publishable_signals_confidence_ge_0.70") or {}
        side = rep.get("side_diagnostics") or {}
        int_ev = (rep.get("backtest_internal_all_signals") or {}).get("expected_ev_usd", 0.0)
        pub_ev = (rep.get("backtest_publishable_confidence_ge_0.70") or {}).get("expected_ev_usd", 0.0)
        summary_rows.append(
            {
                "mode": mode,
                "LONG": internal.get("LONG", 0),
                "SHORT": internal.get("SHORT", 0),
                "SKIP": internal.get("SKIP", 0),
                "pub_LONG_ge_70": pub.get("LONG", 0),
                "pub_SHORT_ge_70": pub.get("SHORT", 0),
                "internal_ev": int_ev,
                "publishable_ev": pub_ev,
                "pub_LONG_ev": (side.get("LONG") or {}).get("publishable_ev", 0.0),
                "pub_SHORT_ev": (side.get("SHORT") or {}).get("publishable_ev", 0.0),
            }
        )

    out = {"ablations": results, "summary_table": summary_rows, "artifacts": str(args.artifacts)}
    compare_modes = ("tabular_only", "no_meta", "two_stage_meta", "side_aware_ensemble")
    out["comparison_table"] = [r for r in summary_rows if r.get("mode") in compare_modes]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.run_id}" if args.run_id else ""
    out_path = args.report_dir / f"ablation{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"summary_table": summary_rows, "report": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
