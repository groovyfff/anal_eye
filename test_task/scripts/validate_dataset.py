#!/usr/bin/env python3
"""Validate canonical dataset integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.training.canonical import CANONICAL_COLUMNS, validate_canonical
from ae_brain.training.labels import compute_labels_for_frame, label_distribution_report
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.training.canonical import candles_from_canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        print("Run: python scripts/download_market_data.py --symbols-from-config ...", file=sys.stderr)
        print("Then: python scripts/prepare_training_dataset.py --symbols-from-config", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.dataset)
    errors = validate_canonical(df)
    report: dict = {"dataset": str(args.dataset), "rows": len(df), "errors": errors}

    if not errors:
        label_frames = []
        for sym in sorted(df["symbol"].unique()):
            sub = df[df["symbol"] == sym].copy()
            candles = candles_from_canonical(sub)
            eng = FeatureEngineer(z_window=100)
            feats = eng.compute_frame(candles)
            atr = feats["atr_14"].to_numpy(float)
            labels, _ = compute_labels_for_frame(candles, atr)
            valid = slice(100, len(labels) - 24)
            label_frames.append(
                pd.DataFrame(
                    {
                        "label": labels[valid],
                        "timestamp": sub["timestamp"].iloc[valid],
                        "symbol": sym,
                    }
                )
            )
        if label_frames:
            lab = pd.concat(label_frames, ignore_index=True)
            report["labels"] = label_distribution_report(
                lab["label"].to_numpy(), lab["timestamp"], lab["symbol"].to_numpy()
            )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "validate_dataset.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
