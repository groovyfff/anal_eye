#!/usr/bin/env python3
"""Top-200 side-balance weight sweep — ranks candidates from summary.json only."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_UNIVERSE_TXT = ROOT / "config" / "universe_top200_usdtm.txt"
DEFAULT_DATASET = ROOT / "data" / "datasets" / "multi_asset.parquet"
DEFAULT_CANDIDATES_DIR = ROOT / "artifacts_candidates"

LONG_POSITIVE_WEIGHTS: tuple[float, ...] = (
    3.05, 3.10, 3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45,
)
SHORT_POSITIVE_WEIGHTS: tuple[float, ...] = (
    1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75,
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "promotable",
    "promotion_blockers",
    "publishable_long_count_ge_70",
    "publishable_short_count_ge_70",
    "publishable_total_trade_count_ge_70",
    "publishable_long_ev_ge_70",
    "publishable_short_ev_ge_70",
    "publishable_total_ev_ge_70",
)


def utc_sweep_id() -> str:
    return f"top200_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def format_weight(value: float) -> str:
    return f"{value:.2f}"


def generate_weight_grid(
    long_weights: tuple[float, ...] = LONG_POSITIVE_WEIGHTS,
    short_weights: tuple[float, ...] = SHORT_POSITIVE_WEIGHTS,
) -> list[tuple[float, float]]:
    return [(float(lw), float(sw)) for lw, sw in product(long_weights, short_weights)]


def candidate_run_id(sweep_id: str, long_weight: float, short_weight: float) -> str:
    return f"{sweep_id}_long_{format_weight(long_weight)}_short_{format_weight(short_weight)}"


def candidate_artifact_dir(
    sweep_id: str,
    long_weight: float,
    short_weight: float,
    *,
    candidates_dir: Path = DEFAULT_CANDIDATES_DIR,
) -> Path:
    return candidates_dir / candidate_run_id(sweep_id, long_weight, short_weight)


def summary_path_for(candidate_dir: Path) -> Path:
    return candidate_dir / "summary.json"


def has_valid_summary(candidate_dir: Path) -> bool:
    path = summary_path_for(candidate_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "promotable" in payload


def parse_summary_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    for key in SUMMARY_FIELDS:
        if key in payload:
            out[key] = payload[key]
    return out


def parse_side_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(report, dict):
        return {}
    threshold = (report.get("second_pass_threshold_report") or {}).get("per_threshold") or {}
    return {
        "diagnostic_only": True,
        "second_pass_threshold_0_70": threshold.get("0.70") or {},
        "calibration_ceiling_summary": report.get("calibration_ceiling_summary"),
        "side_balance": report.get("side_balance"),
    }


def _share_penalty(share: float, lo: float, hi: float) -> float:
    if lo <= share <= hi:
        return 0.0
    if share < lo:
        return lo - share
    return share - hi


@dataclass
class SweepRules:
    min_long_count: int = 30
    min_short_count: int = 30
    min_total_trades: int = 80
    target_long_share_min: float = 0.25
    target_long_share_max: float = 0.45
    target_short_share_min: float = 0.55
    target_short_share_max: float = 0.75


@dataclass
class CandidateResult:
    run_id: str
    long_positive_weight: float
    short_positive_weight: float
    artifact_dir: str
    status: str
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    side_diagnostics: dict[str, Any] = field(default_factory=dict)
    rank_key: tuple[float, ...] | None = None
    long_share: float = 0.0
    short_share: float = 0.0
    imbalance_penalty: float = 0.0


def evaluate_candidate_result(
    *,
    run_id: str,
    long_weight: float,
    short_weight: float,
    artifact_dir: Path,
    rules: SweepRules,
) -> CandidateResult:
    summary_path = summary_path_for(artifact_dir)
    side_path = artifact_dir / "side_specialists_report.json"
    result = CandidateResult(
        run_id=run_id,
        long_positive_weight=long_weight,
        short_positive_weight=short_weight,
        artifact_dir=str(artifact_dir),
        status="failed",
        side_diagnostics=parse_side_diagnostics(side_path),
    )

    summary = parse_summary_json(summary_path)
    if summary is None:
        result.rejection_reasons.append("summary_json_missing")
        return result

    result.summary = summary
    result.status = "completed"

    pub_long = int(summary.get("publishable_long_count_ge_70", 0))
    pub_short = int(summary.get("publishable_short_count_ge_70", 0))
    pub_total = int(summary.get("publishable_total_trade_count_ge_70", pub_long + pub_short))
    pub_long_ev = float(summary.get("publishable_long_ev_ge_70", 0.0))
    pub_short_ev = float(summary.get("publishable_short_ev_ge_70", 0.0))
    pub_total_ev = float(summary.get("publishable_total_ev_ge_70", 0.0))
    promotable = bool(summary.get("promotable", False))

    reasons: list[str] = []
    if not promotable:
        reasons.append("promotable_not_true")
        blockers = summary.get("promotion_blockers") or []
        for blocker in blockers:
            reasons.append(f"blocker:{blocker}")
    if pub_long < rules.min_long_count:
        reasons.append(f"long_count={pub_long}<{rules.min_long_count}")
    if pub_short < rules.min_short_count:
        reasons.append(f"short_count={pub_short}<{rules.min_short_count}")
    if pub_total < rules.min_total_trades:
        reasons.append(f"total_trades={pub_total}<{rules.min_total_trades}")
    if pub_long_ev <= 0:
        reasons.append("long_ev_not_positive")
    if pub_short_ev <= 0:
        reasons.append("short_ev_not_positive")
    if pub_total_ev <= 0:
        reasons.append("total_ev_not_positive")

    long_share = float(pub_long / pub_total) if pub_total else 0.0
    short_share = float(pub_short / pub_total) if pub_total else 0.0
    result.long_share = long_share
    result.short_share = short_share

    long_pen = _share_penalty(long_share, rules.target_long_share_min, rules.target_long_share_max)
    short_pen = _share_penalty(short_share, rules.target_short_share_min, rules.target_short_share_max)
    imbalance = abs(long_share - short_share)
    result.imbalance_penalty = imbalance

    if reasons:
        result.rejection_reasons = reasons
        return result

    result.accepted = True
    result.rank_key = (
        long_pen,
        short_pen,
        imbalance,
        -pub_total_ev,
        -pub_total,
    )
    return result


def rank_candidates(candidates: list[CandidateResult]) -> list[CandidateResult]:
    accepted = [c for c in candidates if c.accepted and c.rank_key is not None]
    rejected = [c for c in candidates if not c.accepted]
    accepted.sort(key=lambda c: c.rank_key or ())
    return accepted + rejected


def build_training_command(
    *,
    run_id: str,
    long_weight: float,
    short_weight: float,
    sample_per_symbol: int,
    batch_size: int,
    resume: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_top200_training.py"),
        "--run-id",
        run_id,
        "--start",
        "2021-01-01",
        "--end",
        "now",
        "--batch-size",
        str(batch_size),
        "--sample-per-symbol",
        str(sample_per_symbol),
        "--meta-mode",
        "side_specialists",
        "--balance-side-specialists",
        "true",
        "--balance-train-samples",
        "true",
        "--allow-skip-sequence",
        "true",
        "--long-positive-weight",
        format_weight(long_weight),
        "--short-positive-weight",
        format_weight(short_weight),
        "--universe-txt",
        str(DEFAULT_UNIVERSE_TXT),
        "--dataset",
        str(DEFAULT_DATASET),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def _run_training_job(payload: dict[str, Any]) -> dict[str, Any]:
    cmd = payload["cmd"]
    run_id = payload["run_id"]
    artifact_dir = Path(payload["artifact_dir"])
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "returncode": proc.returncode,
        "summary_exists": has_valid_summary(artifact_dir),
    }


def run_candidate_training(
    *,
    run_id: str,
    long_weight: float,
    short_weight: float,
    sample_per_symbol: int,
    batch_size: int,
    resume: bool,
) -> int:
    cmd = build_training_command(
        run_id=run_id,
        long_weight=long_weight,
        short_weight=short_weight,
        sample_per_symbol=sample_per_symbol,
        batch_size=batch_size,
        resume=resume,
    )
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def write_sweep_summary(
    *,
    sweep_id: str,
    candidates: list[CandidateResult],
    output_dir: Path,
    rules: SweepRules,
    grid_size: int,
    executed: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_candidates(candidates)
    accepted = [c for c in ranked if c.accepted]
    payload = {
        "sweep_id": sweep_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_source_of_truth": "summary.json",
        "side_specialists_report_diagnostic_only": True,
        "grid_size": grid_size,
        "executed_runs": executed,
        "accepted_count": len(accepted),
        "rules": asdict(rules),
        "best_candidate": asdict(accepted[0]) if accepted else None,
        "ranked_candidates": [asdict(c) for c in ranked],
    }
    json_path = output_dir / f"{sweep_id}_summary.json"
    csv_path = output_dir / f"{sweep_id}_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "rank",
        "accepted",
        "run_id",
        "long_positive_weight",
        "short_positive_weight",
        "status",
        "promotable",
        "publishable_long_count_ge_70",
        "publishable_short_count_ge_70",
        "publishable_total_trade_count_ge_70",
        "publishable_long_ev_ge_70",
        "publishable_short_ev_ge_70",
        "publishable_total_ev_ge_70",
        "long_share",
        "short_share",
        "imbalance_penalty",
        "rejection_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        rank = 0
        for cand in ranked:
            if cand.accepted:
                rank += 1
            summary = cand.summary or {}
            writer.writerow(
                {
                    "rank": rank if cand.accepted else "",
                    "accepted": cand.accepted,
                    "run_id": cand.run_id,
                    "long_positive_weight": cand.long_positive_weight,
                    "short_positive_weight": cand.short_positive_weight,
                    "status": cand.status,
                    "promotable": summary.get("promotable"),
                    "publishable_long_count_ge_70": summary.get("publishable_long_count_ge_70"),
                    "publishable_short_count_ge_70": summary.get("publishable_short_count_ge_70"),
                    "publishable_total_trade_count_ge_70": summary.get("publishable_total_trade_count_ge_70"),
                    "publishable_long_ev_ge_70": summary.get("publishable_long_ev_ge_70"),
                    "publishable_short_ev_ge_70": summary.get("publishable_short_ev_ge_70"),
                    "publishable_total_ev_ge_70": summary.get("publishable_total_ev_ge_70"),
                    "long_share": cand.long_share,
                    "short_share": cand.short_share,
                    "imbalance_penalty": cand.imbalance_penalty,
                    "rejection_reasons": ";".join(cand.rejection_reasons),
                }
            )
    return json_path, csv_path


def should_skip_candidate(
    artifact_dir: Path,
    *,
    retry_failed: bool,
) -> bool:
    if has_valid_summary(artifact_dir):
        return True
    if retry_failed:
        return False
    return artifact_dir.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-200 side-balance weight sweep runner")
    parser.add_argument("--sweep-id", default="")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--sample-per-symbol", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--min-long-count", type=int, default=30)
    parser.add_argument("--min-short-count", type=int, default=30)
    parser.add_argument("--min-total-trades", type=int, default=80)
    parser.add_argument("--target-long-share-min", type=float, default=0.25)
    parser.add_argument("--target-long-share-max", type=float, default=0.45)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan grid and ranking only; do not train")
    args = parser.parse_args()

    sweep_id = args.sweep_id or utc_sweep_id()
    rules = SweepRules(
        min_long_count=args.min_long_count,
        min_short_count=args.min_short_count,
        min_total_trades=args.min_total_trades,
        target_long_share_min=args.target_long_share_min,
        target_long_share_max=args.target_long_share_max,
    )
    grid = generate_weight_grid()
    pending: list[tuple[float, float, str, Path]] = []
    for long_weight, short_weight in grid:
        run_id = candidate_run_id(sweep_id, long_weight, short_weight)
        artifact_dir = candidate_artifact_dir(sweep_id, long_weight, short_weight)
        if should_skip_candidate(artifact_dir, retry_failed=args.retry_failed):
            continue
        pending.append((long_weight, short_weight, run_id, artifact_dir))

    if args.max_runs is not None:
        pending = pending[: max(0, args.max_runs)]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "sweep_id": sweep_id,
                    "grid_size": len(grid),
                    "pending_runs": len(pending),
                    "pending_run_ids": [item[2] for item in pending],
                },
                indent=2,
            )
        )
        return

    executed = 0
    if pending:
        if args.parallel <= 1:
            for long_weight, short_weight, run_id, artifact_dir in pending:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                run_candidate_training(
                    run_id=run_id,
                    long_weight=long_weight,
                    short_weight=short_weight,
                    sample_per_symbol=args.sample_per_symbol,
                    batch_size=args.batch_size,
                    resume=args.retry_failed and artifact_dir.exists(),
                )
                executed += 1
        else:
            jobs = []
            for long_weight, short_weight, run_id, artifact_dir in pending:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    {
                        "run_id": run_id,
                        "artifact_dir": str(artifact_dir),
                        "cmd": build_training_command(
                            run_id=run_id,
                            long_weight=long_weight,
                            short_weight=short_weight,
                            sample_per_symbol=args.sample_per_symbol,
                            batch_size=args.batch_size,
                            resume=args.retry_failed and artifact_dir.exists(),
                        ),
                    }
                )
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = [pool.submit(_run_training_job, job) for job in jobs]
                for fut in as_completed(futures):
                    fut.result()
                    executed += 1

    candidates: list[CandidateResult] = []
    for long_weight, short_weight in grid:
        run_id = candidate_run_id(sweep_id, long_weight, short_weight)
        artifact_dir = candidate_artifact_dir(sweep_id, long_weight, short_weight)
        candidates.append(
            evaluate_candidate_result(
                run_id=run_id,
                long_weight=long_weight,
                short_weight=short_weight,
                artifact_dir=artifact_dir,
                rules=rules,
            )
        )

    json_path, csv_path = write_sweep_summary(
        sweep_id=sweep_id,
        candidates=candidates,
        output_dir=DEFAULT_CANDIDATES_DIR,
        rules=rules,
        grid_size=len(grid),
        executed=executed,
    )
    accepted = [c for c in candidates if c.accepted]
    print(f"Sweep complete: {sweep_id}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Accepted candidates: {len(accepted)} / {len(grid)}")
    if accepted:
        best = rank_candidates(candidates)[0]
        print(
            "Best candidate:",
            best.run_id,
            f"LONG={best.summary.get('publishable_long_count_ge_70')}",
            f"SHORT={best.summary.get('publishable_short_count_ge_70')}",
            f"EV={best.summary.get('publishable_total_ev_ge_70')}",
        )
    print("Promotion is manual only; this sweep never auto-promotes.")


if __name__ == "__main__":
    main()
