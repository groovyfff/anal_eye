#!/usr/bin/env python3
"""Promote a top-200 candidate artifact to production with backup."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.artifact_verify import verify_runtime_artifacts
from ae_brain.training.promotion import evaluate_promotion, save_promotion_report, verify_artifacts_match


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_production_artifacts(production_dir: Path, backups_root: Path) -> Path:
    backup_dir = backups_root / _utc_run_id()
    backup_dir.mkdir(parents=True, exist_ok=True)
    if production_dir.exists() and any(production_dir.iterdir()):
        for item in production_dir.iterdir():
            dest = backup_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    return backup_dir


def promote_with_backup(
    candidate_dir: Path,
    production_dir: Path,
    backups_root: Path,
    *,
    force: bool = False,
) -> Path:
    summary_path = candidate_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    metrics_path = candidate_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = evaluate_promotion(metrics, summary=summary)
    save_promotion_report(result, candidate_dir / "promotion_result.json")
    if not result.passed and not force:
        raise RuntimeError(f"Promotion rejected: {result.reasons}")

    backup_dir = backup_production_artifacts(production_dir, backups_root)
    staging = production_dir.parent / f".promote_staging_{candidate_dir.name}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(candidate_dir, staging)
        if production_dir.exists():
            shutil.rmtree(production_dir)
        shutil.move(str(staging), str(production_dir))
        verify_artifacts_match(candidate_dir, production_dir)
        verify_runtime_artifacts(production_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if not production_dir.exists() and backup_dir.exists():
            shutil.copytree(backup_dir, production_dir)
        raise
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote top-200 candidate artifacts to production")
    parser.add_argument("--run-id", required=True, help="Candidate run id under artifacts_candidates/")
    parser.add_argument("--candidates-dir", type=Path, default=ROOT / "artifacts_candidates")
    parser.add_argument("--production-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--backups-dir", type=Path, default=ROOT / "artifacts_backups")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidate = args.candidates_dir / args.run_id
    if not candidate.is_dir():
        print(f"Candidate not found: {candidate}", file=sys.stderr)
        sys.exit(1)

    try:
        backup_dir = promote_with_backup(
            candidate,
            args.production_dir,
            args.backups_dir,
            force=args.force,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"Promoted {candidate} -> {args.production_dir}")
    print(f"Backup stored at {backup_dir}")
    print("Runtime artifact verification passed.")


if __name__ == "__main__":
    main()
