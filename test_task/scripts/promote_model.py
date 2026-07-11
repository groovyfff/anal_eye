#!/usr/bin/env python3
"""Promote candidate artifacts to production after promotion rules pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.training.promotion import evaluate_promotion, promote_artifacts, save_promotion_report, verify_artifacts_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates-dir", type=Path, default=ROOT / "artifacts_candidates")
    parser.add_argument("--production-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--metrics", type=Path, help="backtest/metrics JSON for promotion rules")
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidate = args.candidates_dir / args.run_id
    if not candidate.is_dir():
        print(f"Candidate not found: {candidate}", file=sys.stderr)
        sys.exit(1)

    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    metrics_path = args.metrics
    if metrics_path is None:
        metrics_path = candidate / "test_metrics.json"
    if not metrics_path.exists():
        print(f"Missing test metrics: {metrics_path}", file=sys.stderr)
        print("Run: python scripts/evaluate_candidate.py --run-id", args.run_id, file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text())
    if not metrics:
        print("test_metrics.json is empty — run evaluate_candidate first.", file=sys.stderr)
        sys.exit(1)

    baseline = None
    if args.baseline_metrics and args.baseline_metrics.exists():
        baseline = json.loads(args.baseline_metrics.read_text())

    result = evaluate_promotion(metrics, baseline_metrics=baseline, summary=summary)
    report_path = candidate / "promotion_result.json"
    save_promotion_report(result, report_path)

    if not result.passed and not args.force:
        print("Promotion REJECTED:", result.reasons, file=sys.stderr)
        sys.exit(1)

    backup_path = promote_artifacts(candidate, args.production_dir)
    print(f"Promoted {candidate} -> {args.production_dir}")
    if backup_path is not None:
        print(f"Previous artifacts backed up to {backup_path}")
    verify_artifacts_match(candidate, args.production_dir)
    print("Promotion sync verified: candidate and production trees match.")


if __name__ == "__main__":
    main()
