"""Model promotion rules — never overwrite production artifacts without passing gates."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ae_brain.training.evaluation import PUBLISHABLE_GE_70_FIELD_NAMES


@dataclass
class PromotionRules:
    min_test_ev_usd: float = 0.0
    min_test_pnl_usd: float = 0.0
    max_drawdown: float = 0.35
    min_trades: int = 20
    min_long_count: int = 1
    min_short_count: int = 1
    min_publishable_long: int = 1
    min_publishable_short: int = 1
    min_precision_at_70: float = 0.45
    max_calibration_brier: float = 0.35
    max_symbol_pnl_share: float = 0.60
    require_positive_ev_at_70: bool = True
    publish_confidence: float = 0.70
    require_test_metrics_file: bool = True
    # Minimum share each side must hold of the publishable trade count. A model
    # that is heavily one-sided (e.g. SHORT-only) cannot be promoted even with
    # positive EV. Default 0.10 (10%) - exposed in the report for diagnostics.
    min_publishable_long_share: float = 0.10
    min_publishable_short_share: float = 0.10


@dataclass
class PromotionResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_promotion(
    test_metrics: dict[str, Any],
    *,
    baseline_metrics: dict[str, Any] | None = None,
    rules: PromotionRules | None = None,
    summary: dict[str, Any] | None = None,
) -> PromotionResult:
    rules = rules or PromotionRules()
    reasons: list[str] = []

    if not test_metrics:
        reasons.append("test_metrics_missing_or_empty")

    missing_side_fields = [f for f in PUBLISHABLE_GE_70_FIELD_NAMES if f not in test_metrics]
    if missing_side_fields:
        reasons.append("publishable_side_metrics_missing")

    ev = float(test_metrics.get("expected_ev_usd", 0.0))
    pnl = float(test_metrics.get("net_pnl_usd", 0.0))
    dd = float(test_metrics.get("max_drawdown", 1.0))
    trades = int(test_metrics.get("trade_count", 0))
    long_c = int(test_metrics.get("long_count", 0))
    short_c = int(test_metrics.get("short_count", 0))
    prec70 = test_metrics.get("precision_at_conf_70")

    internal = test_metrics.get("internal_model_signals") or {}
    internal_long = int(internal.get("LONG", long_c))
    internal_short = int(internal.get("SHORT", short_c))
    if internal_long == 0:
        reasons.append("internal_long_count_zero")
    if internal_short == 0:
        reasons.append("internal_short_count_zero")

    publishable = test_metrics.get("publishable_signals_ge_70")
    if publishable is None and summary is not None:
        publishable = summary.get("publishable_signals_ge_70", {})
    publishable = publishable or {}
    pub_long = int(test_metrics.get("publishable_long_count_ge_70", publishable.get("LONG", 0)))
    pub_short = int(test_metrics.get("publishable_short_count_ge_70", publishable.get("SHORT", 0)))
    pub_long_ev = test_metrics.get("publishable_long_ev_ge_70")
    pub_short_ev = test_metrics.get("publishable_short_ev_ge_70")

    if long_c < rules.min_long_count:
        reasons.append(f"long_count={long_c} < {rules.min_long_count}")
    if short_c < rules.min_short_count:
        reasons.append(f"short_count={short_c} < {rules.min_short_count}")
    if long_c == 0 or short_c == 0:
        reasons.append("model_is_one_sided")
    if pub_long < rules.min_publishable_long:
        reasons.append(f"publishable_long_at_{rules.publish_confidence:.2f}={pub_long}")
    if pub_short < rules.min_publishable_short:
        reasons.append(f"publishable_short_at_{rules.publish_confidence:.2f}={pub_short}")
    if pub_long + pub_short == 0:
        reasons.append(f"publishable_signals_at_{rules.publish_confidence:.2f}_zero")

    pub_total = pub_long + pub_short
    side_balance_metrics: dict[str, Any] = {}
    if pub_total > 0:
        long_share = pub_long / pub_total
        short_share = pub_short / pub_total
        side_balance_metrics = {
            "publishable_long_share": float(long_share),
            "publishable_short_share": float(short_share),
            "min_required_long_share": float(rules.min_publishable_long_share),
            "min_required_short_share": float(rules.min_publishable_short_share),
            "publishable_long_count": pub_long,
            "publishable_short_count": pub_short,
            "publishable_total": pub_total,
        }
        if pub_long > 0 and long_share < rules.min_publishable_long_share:
            reasons.append(f"publishable_long_share_too_low={long_share:.3f}")
        if pub_short > 0 and short_share < rules.min_publishable_short_share:
            reasons.append(f"publishable_short_share_too_low={short_share:.3f}")

    if not missing_side_fields:
        if pub_long > 0 and float(pub_long_ev) <= 0:
            reasons.append("publishable_long_ev_not_positive")
        if pub_short > 0 and float(pub_short_ev) <= 0:
            reasons.append("publishable_short_ev_not_positive")
        pub_total_ev = float(test_metrics.get("publishable_total_ev_ge_70", 0.0))
        if pub_total > 0 and pub_total_ev <= 0:
            reasons.append("publishable_ev_not_positive")

    pub_backtest = test_metrics.get("backtest_publishable_ge_70") or {}
    if summary is not None and not pub_backtest:
        pub_backtest = summary.get("publishable_backtest_ge_70") or {}
    if int(pub_backtest.get("trade_count", 0)) == 0 and pub_long + pub_short == 0:
        reasons.append(f"publishable_backtest_no_trades_at_{rules.publish_confidence:.2f}")

    if ev <= rules.min_test_ev_usd:
        reasons.append(f"test_ev_usd={ev} <= {rules.min_test_ev_usd}")
    if pnl <= rules.min_test_pnl_usd:
        reasons.append(f"test_pnl_usd={pnl} <= {rules.min_test_pnl_usd}")
    if baseline_metrics is not None:
        if ev <= float(baseline_metrics.get("expected_ev_usd", ev)):
            reasons.append("test_ev_not_above_baseline")
        if pnl <= float(baseline_metrics.get("net_pnl_usd", pnl)):
            reasons.append("test_pnl_not_above_baseline")
    if dd > rules.max_drawdown:
        reasons.append(f"max_drawdown={dd} > {rules.max_drawdown}")
    if trades < rules.min_trades:
        reasons.append(f"trade_count={trades} < {rules.min_trades}")
    if prec70 is not None and float(prec70) < rules.min_precision_at_70:
        reasons.append(f"precision_at_70={prec70} < {rules.min_precision_at_70}")
    if prec70 is None and pub_long + pub_short == 0:
        reasons.append("precision_at_conf_70_unavailable_no_publishable_trades")

    if summary is not None:
        cal = (
            (summary.get("training_metrics") or {}).get("meta", {}).get("calibration")
            or (summary.get("calibration") or {})
        )
        side_cals = cal.get("LONG") or cal.get("long") or {}
        if isinstance(cal, dict) and "LONG" in cal and "SHORT" in cal:
            for side in ("LONG", "SHORT"):
                sc = cal.get(side) or {}
                if sc.get("error"):
                    reasons.append(f"calibration_missing_{side}:{sc.get('error')}")
                brier = sc.get("brier_calibrated")
                if brier is not None and float(brier) > rules.max_calibration_brier:
                    reasons.append(f"calibration_brier_{side}_too_high={brier}")
        else:
            brier_cal = cal.get("brier_calibrated") if isinstance(cal, dict) else None
            if brier_cal is not None and float(brier_cal) > rules.max_calibration_brier:
                reasons.append(f"calibration_brier_too_high={brier_cal}")
            if isinstance(cal, dict) and cal.get("error"):
                reasons.append(f"calibration_missing:{cal.get('error')}")

    per_sym = test_metrics.get("per_symbol") or {}
    total_abs = sum(abs(v.get("net_pnl_usd", 0.0)) for v in per_sym.values()) or 1.0
    for sym, stats in per_sym.items():
        share = abs(stats.get("net_pnl_usd", 0.0)) / total_abs
        if share > rules.max_symbol_pnl_share:
            reasons.append(f"symbol_dominance {sym} share={share:.2f}")
    if rules.require_positive_ev_at_70:
        buckets = test_metrics.get("ev_by_confidence_bucket") or {}
        hi = buckets.get("0.70-0.80", buckets.get("0.70-1.01", 0.0))
        if hi is not None and float(hi) <= 0:
            reasons.append("ev_at_conf_70_not_positive")

    deduped: list[str] = []
    for r in reasons:
        if r not in deduped:
            deduped.append(r)
    merged_metrics = dict(test_metrics)
    if side_balance_metrics:
        merged_metrics["side_balance"] = side_balance_metrics
    return PromotionResult(passed=len(deduped) == 0, reasons=deduped, metrics=merged_metrics)


def verify_artifacts_match(candidate_dir: Path, production_dir: Path) -> None:
    """Raise if production artifacts differ from the promoted candidate tree."""
    import subprocess

    result = subprocess.run(
        ["diff", "-qr", str(candidate_dir.resolve()), str(production_dir.resolve())],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = f"{result.stdout}{result.stderr}".strip()
        raise RuntimeError(f"Promotion sync verification failed; diff not empty:\n{details}")


def promote_artifacts(candidate_dir: Path, production_dir: Path, *, backup: bool = True) -> Path | None:
    """Atomically replace production artifacts with candidate contents (no stale files)."""
    candidate_dir = candidate_dir.resolve()
    production_dir = production_dir.resolve()
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)

    backup_path: Path | None = None
    if backup and production_dir.exists() and any(production_dir.iterdir()):
        backup_path = production_dir.parent / f"artifacts_backup_{candidate_dir.name}"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(production_dir, backup_path)

    staging = production_dir.parent / f".promote_staging_{candidate_dir.name}"
    if staging.exists():
        shutil.rmtree(staging)

    try:
        shutil.copytree(candidate_dir, staging)
        if production_dir.exists():
            shutil.rmtree(production_dir)
        shutil.move(str(staging), str(production_dir))
        verify_artifacts_match(candidate_dir, production_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return backup_path


def save_promotion_report(result: PromotionResult, path: Path) -> None:
    path.write_text(
        json.dumps({"passed": result.passed, "reasons": result.reasons, "metrics": result.metrics}, indent=2)
    )
