#!/usr/bin/env python3
"""Resumable, memory-safe top-200 training orchestrator for AE Brain."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.universe_top200 import LEGACY_SIX_SYMBOLS, load_universe_txt, symbols_csv
from scripts.download_market_data import (
    TimeRange,
    download_symbol,
    parse_cli_time_range,
    timestamp_to_ms,
)
from scripts.download_market_data import _read_csv_timestamps

DEFAULT_UNIVERSE_TXT = ROOT / "config" / "universe_top200_usdtm.txt"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "binance"
DEFAULT_DATASET = ROOT / "data" / "datasets" / "multi_asset.parquet"
PUBLISH_CONFIDENCE = 0.70


def utc_run_id(prefix: str = "top200") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("top200_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def klines_complete(path: Path, time_range: TimeRange, *, min_rows: int = 100) -> bool:
    if not path.exists():
        return False
    try:
        df = _read_csv_timestamps(path)
    except Exception:
        return False
    if len(df) < min_rows:
        return False
    last_ms = timestamp_to_ms(df["timestamp"].iloc[-1])
    # Up-to-date if the last closed 1h candle is within two hours of the requested end.
    return last_ms >= time_range.end_ms - (2 * 3_600_000)


def download_symbols_batched(
    symbols: list[str],
    *,
    raw_dir: Path,
    time_range: TimeRange,
    batch_size: int,
    state: dict[str, Any],
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    downloaded = set(state.get("downloaded_symbols") or [])
    failures: list[str] = []
    pending = [s for s in symbols if s not in downloaded or not klines_complete(raw_dir / s / "1h" / "klines.csv", time_range)]
    logger.info("download.pending=%s already_complete=%s", len(pending), len(symbols) - len(pending))

    for idx in range(0, len(pending), batch_size):
        batch = pending[idx : idx + batch_size]
        logger.info("download.batch start=%s size=%s", idx, len(batch))
        if dry_run:
            continue
        for sym in batch:
            out_path = raw_dir / sym / "1h" / "klines.csv"
            if klines_complete(out_path, time_range):
                logger.info("download.skip symbol=%s reason=complete", sym)
                downloaded.add(sym)
                continue
            try:
                download_symbol(
                    sym,
                    "1h",
                    raw_dir,
                    time_range=time_range,
                    include_funding=True,
                    include_mark=False,
                    include_index=False,
                    include_oi=False,
                )
                downloaded.add(sym)
                logger.info("download.ok symbol=%s", sym)
            except Exception as exc:
                logger.error("download.fail symbol=%s err=%s", sym, exc)
                failures.append(sym)
        state["downloaded_symbols"] = sorted(downloaded)

    if failures and not dry_run:
        raise RuntimeError(f"download failures: {failures[:10]}{'...' if len(failures) > 10 else ''}")


def build_publishable_report(test_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "publish_confidence": PUBLISH_CONFIDENCE,
        "publishable_long_count_ge_70": int(test_metrics.get("publishable_long_count_ge_70", 0)),
        "publishable_short_count_ge_70": int(test_metrics.get("publishable_short_count_ge_70", 0)),
        "publishable_long_ev_ge_70": float(test_metrics.get("publishable_long_ev_ge_70", 0.0)),
        "publishable_short_ev_ge_70": float(test_metrics.get("publishable_short_ev_ge_70", 0.0)),
        "publishable_total_ev_ge_70": float(test_metrics.get("publishable_total_ev_ge_70", 0.0)),
        "publishable_total_trade_count_ge_70": int(
            test_metrics.get("publishable_total_trade_count_ge_70", 0)
        ),
        "expected_ev_usd": float(test_metrics.get("expected_ev_usd", 0.0)),
    }


def warn_publishable_sides(test_metrics: dict[str, Any], *, logger: logging.Logger) -> list[str]:
    """Log side imbalance; never raise — summary.json is the promotion source of truth."""
    warnings: list[str] = []
    pub_long = int(test_metrics.get("publishable_long_count_ge_70", 0))
    pub_short = int(test_metrics.get("publishable_short_count_ge_70", 0))
    if pub_long <= 0:
        msg = "no publishable LONG signals on test split at confidence >= 0.70"
        warnings.append(msg)
        logger.warning(msg)
    if pub_short <= 0:
        msg = "no publishable SHORT signals on test split at confidence >= 0.70"
        warnings.append(msg)
        logger.warning(msg)
    return warnings


def run_subprocess(
    cmd: list[str],
    *,
    logger: logging.Logger,
    cwd: Path = ROOT,
    allow_exit_codes: tuple[int, ...] = (),
) -> int:
    logger.info("exec %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0 and result.returncode not in allow_exit_codes:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def _side_specialists_diagnostic_snapshot(artifact_dir: Path) -> dict[str, Any] | None:
    """Extract validation-only diagnostics; must not drive promotion decisions."""
    report_path = artifact_dir / "side_specialists_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = (report.get("second_pass_threshold_report") or {}).get("per_threshold") or {}
    at_70 = threshold.get("0.70") or {}
    return {
        "diagnostic_only": True,
        "promotion_source_of_truth": "summary.json",
        "note": (
            "side_specialists_report reflects validation-calibration specialist metrics; "
            "promotion uses summary.json publishable_* fields from the held-out test backtest at 0.70."
        ),
        "second_pass_threshold_0_70": at_70,
        "calibration_ceiling_summary": report.get("calibration_ceiling_summary"),
        "side_balance": report.get("side_balance"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-200 AE Brain training orchestrator")
    parser.add_argument("--universe-txt", type=Path, default=DEFAULT_UNIVERSE_TXT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="now")
    parser.add_argument("--sample-per-symbol", type=int, default=12_000)
    parser.add_argument("--meta-mode", default="side_specialists")
    parser.add_argument("--balance-side-specialists", default="true", choices=["true", "false"],
                        help="Enable class balancing for side specialists (default true for top200)")
    parser.add_argument("--long-positive-weight", default="auto",
                        help="LONG profitable-class weight: 'auto' (neg/pos imbalance) or a float")
    parser.add_argument("--short-positive-weight", default="auto",
                        help="SHORT profitable-class weight: 'auto' (neg/pos imbalance) or a float")
    parser.add_argument("--balance-train-samples", default="true", choices=["true", "false"],
                        help="Balance (undersample) the majority class within the train split only (default true)")
    parser.add_argument("--max-side-train-samples-per-class", type=int, default=None)
    parser.add_argument("--allow-skip-sequence", default="true", choices=["true", "false"],
                        help="Memory-safe: skip sequence training if too heavy/fails (default true for top200)")
    parser.add_argument("--skip-sequence", default="true", choices=["true", "false"],
                        help="Proactively skip PatchTST sequence training (default true for top200)")
    parser.add_argument("--skip-rl", default="true", choices=["true", "false"],
                        help="Proactively skip PPO RL training (default true for top200)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    symbols = load_universe_txt(args.universe_txt)
    if len(symbols) != 200:
        print(f"Expected 200 symbols in {args.universe_txt}, got {len(symbols)}", file=sys.stderr)
        sys.exit(1)
    for legacy in LEGACY_SIX_SYMBOLS:
        if legacy not in symbols:
            print(f"Legacy symbol missing from universe: {legacy}", file=sys.stderr)
            sys.exit(1)

    run_id = args.run_id or utc_run_id()
    artifact_dir = ROOT / "artifacts_candidates" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / "state.json"
    state = load_state(state_path) if args.resume else {}
    state.setdefault("run_id", run_id)
    state.setdefault("symbols", symbols)
    logger = setup_logger(artifact_dir / "logs" / "train.log")

    (artifact_dir / "config").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config" / "universe_snapshot.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")

    time_range = parse_cli_time_range(args.start, args.end)
    logger.info("run_id=%s symbols=%s dry_run=%s", run_id, len(symbols), args.dry_run)

    if args.dry_run:
        logger.info(
            "DRY RUN plan: download=%s prepare_dataset train evaluate reports artifact_dir=%s",
            not args.skip_download,
            artifact_dir,
        )
        return

    if not args.skip_download:
        download_symbols_batched(
            symbols,
            raw_dir=args.raw_dir,
            time_range=time_range,
            batch_size=args.batch_size,
            state=state,
            logger=logger,
            dry_run=False,
        )
        save_state(state_path, state)

    if not state.get("dataset_prepared"):
        run_subprocess(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_training_dataset.py"),
                "--symbols",
                symbols_csv(symbols),
                "--timeframes",
                "1h",
                "--input",
                str(args.raw_dir),
                "--output",
                str(args.dataset),
            ],
            logger=logger,
        )
        state["dataset_prepared"] = True
        save_state(state_path, state)

    if not args.skip_train and not state.get("training_done"):
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_multi_asset.py"),
            "--dataset",
            str(args.dataset),
            "--symbols",
            symbols_csv(symbols),
            "--interval",
            "1h",
            "--output-dir",
            str(artifact_dir),
            "--meta-mode",
            args.meta_mode,
            "--medium",
            "--sample-per-symbol",
            str(args.sample_per_symbol),
            "--balance-side-specialists",
            args.balance_side_specialists,
            "--long-positive-weight",
            args.long_positive_weight,
            "--short-positive-weight",
            args.short_positive_weight,
            "--balance-train-samples",
            args.balance_train_samples,
            "--allow-skip-sequence",
            args.allow_skip_sequence,
            "--skip-sequence",
            args.skip_sequence,
            "--skip-rl",
            args.skip_rl,
            "--skip-evaluate",
        ]
        if args.max_side_train_samples_per_class is not None:
            train_cmd.extend([
                "--max-side-train-samples-per-class",
                str(args.max_side_train_samples_per_class),
            ])
        run_subprocess(train_cmd, logger=logger)
        state["training_done"] = True
        save_state(state_path, state)

    summary_path = artifact_dir / "summary.json"
    if not state.get("evaluation_done"):
        exit_code = run_subprocess(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_candidate.py"),
                "--run-id",
                run_id,
                "--dataset",
                str(args.dataset),
                "--symbols",
                symbols_csv(symbols),
                "--publish-confidence",
                str(PUBLISH_CONFIDENCE),
            ],
            logger=logger,
            allow_exit_codes=(2,),
        )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"evaluate_candidate exited {exit_code} but summary.json is missing: {summary_path}"
            )
        state["evaluation_done"] = True
        save_state(state_path, state)

    test_metrics_path = artifact_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {test_metrics_path}")
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    side_warnings = warn_publishable_sides(test_metrics, logger=logger)

    from scripts.diagnose_generalization_gap import build_generalization_report

    gen_report = build_generalization_report(run_id, artifact_dir, args.dataset, symbols)
    pub_report = build_publishable_report(test_metrics)
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generalization_report.json").write_text(json.dumps(gen_report, indent=2), encoding="utf-8")
    (reports_dir / "publishable_report.json").write_text(json.dumps(pub_report, indent=2), encoding="utf-8")

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    training_summary_path = artifact_dir / "training_summary.json"
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary.get("sequence_skipped"):
            summary["sequence_skipped"] = True
        if training_summary.get("rl_skipped"):
            summary["rl_skipped"] = True
    side_diag = _side_specialists_diagnostic_snapshot(artifact_dir)
    summary["top200_pipeline"] = {
        "run_id": run_id,
        "symbol_count": len(symbols),
        "publishable_report": pub_report,
        "reports_dir": str(reports_dir),
        "promotion_source_of_truth": "summary.json",
        "side_specialists_diagnostic_only": side_diag is not None,
    }
    if side_warnings:
        summary.setdefault("warnings", [])
        for w in side_warnings:
            if w not in summary["warnings"]:
                summary["warnings"].append(w)
    if side_diag is not None:
        summary["side_specialists_diagnostic"] = side_diag
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("top200 training complete run_id=%s artifact_dir=%s", run_id, artifact_dir)
    print(f"Artifacts: {artifact_dir}")
    print(f"Promote: python scripts/promote_top200_artifact.py --run-id {run_id}")


if __name__ == "__main__":
    main()
