"""Tests for evaluation reporting and promotion blockers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ae_brain.training.evaluation import SignalBatch, assess_run_health, build_evaluation_report
from ae_brain.training.promotion import PromotionRules, evaluate_promotion, promote_artifacts, verify_artifacts_match


def _batch(
    decisions: list[str],
    confs: list[float],
    publishable: list[bool] | None = None,
    symbols: list[str] | None = None,
) -> SignalBatch:
    n = len(decisions)
    if symbols is None:
        symbols = ["BTCUSDT" if i % 2 == 0 else "ETHUSDT" for i in range(n)]
    return SignalBatch(
        decisions=np.array(decisions),
        expected_evs=np.array([100.0 if d != "SKIP" else 0.0 for d in decisions]),
        confidence=np.array(confs, dtype=float),
        symbols=np.array(symbols),
        timestamps=np.array([f"2025-01-{i+1:02d}T00:00:00+00:00" for i in range(n)]),
        publishable=np.array(publishable if publishable is not None else [False] * n, dtype=bool),
    )


def test_assess_run_health_blocks_one_sided_and_zero_publishable() -> None:
    batch = _batch(["SHORT"] * 10 + ["SKIP"] * 5, [0.4] * 15, [False] * 15)
    report = build_evaluation_report(batch, publish_confidence=0.70)
    assert report["promotable"] is False
    assert "internal_long_count_zero" in report["promotion_blockers"]
    assert "publishable_signals_at_0.70_zero" in report["promotion_blockers"]


def test_assess_run_health_passes_balanced_publishable() -> None:
    decisions = ["LONG", "SHORT"] * 12 + ["SKIP"]
    confs = [0.75, 0.80] * 12 + [0.2]
    publishable = [True, True] * 12 + [False]
    batch = _batch(decisions, confs, publishable)
    report = build_evaluation_report(batch, publish_confidence=0.70)
    assert report["promotable"] is True
    assert report["publishable_long_ev_ge_70"] > 0
    assert report["publishable_short_ev_ge_70"] > 0


def test_promotion_rejects_missing_metrics() -> None:
    result = evaluate_promotion({})
    assert result.passed is False
    assert "test_metrics_missing_or_empty" in result.reasons


def test_promotion_rejects_short_only_with_summary(tmp_path: Path) -> None:
    metrics = {
        "expected_ev_usd": 1000.0,
        "net_pnl_usd": 1000.0,
        "max_drawdown": 0.1,
        "trade_count": 100,
        "long_count": 0,
        "short_count": 100,
        "precision_at_conf_70": None,
        "per_symbol": {"BTCUSDT": {"net_pnl_usd": 1000.0, "trades": 100}},
        "ev_by_confidence_bucket": {"0.70-0.80": 0.0},
    }
    summary = {
        "publishable_signals_ge_70": {"LONG": 0, "SHORT": 0},
        "publishable_backtest_ge_70": {"trade_count": 0},
        "promotion_blockers": ["internal_long_count_zero"],
    }
    result = evaluate_promotion(metrics, summary=summary)
    assert result.passed is False
    assert any("long_count=0" in r or "model_is_one_sided" in r for r in result.reasons)


def test_promotion_rejects_missing_side_ev_fields_despite_positive_total() -> None:
    """Regression: total publishable EV > 0 but side-level fields absent -> not promotable."""
    metrics = {
        "expected_ev_usd": 5000.0,
        "net_pnl_usd": 5000.0,
        "max_drawdown": 0.1,
        "trade_count": 50,
        "long_count": 25,
        "short_count": 25,
        "precision_at_conf_70": 0.8,
        "per_symbol": {"BTCUSDT": {"net_pnl_usd": 5000.0, "trades": 50}},
        "ev_by_confidence_bucket": {"0.70-0.80": 100.0},
        "internal_model_signals": {"LONG": 25, "SHORT": 25, "SKIP": 0},
        "publishable_signals_ge_70": {"LONG": 10, "SHORT": 10},
        "backtest_publishable_ge_70": {"trade_count": 20, "net_pnl_usd": 5000.0},
    }
    summary = {
        "publishable_signals_ge_70": {"LONG": 10, "SHORT": 10},
        "publishable_backtest_ge_70": {"trade_count": 20, "net_pnl_usd": 5000.0},
    }
    result = evaluate_promotion(metrics, summary=summary)
    assert result.passed is False
    assert "publishable_side_metrics_missing" in result.reasons


def test_promotion_passes_with_explicit_side_ev_fields() -> None:
    metrics = {
        "expected_ev_usd": 3000.0,
        "net_pnl_usd": 3000.0,
        "max_drawdown": 0.1,
        "trade_count": 20,
        "long_count": 10,
        "short_count": 10,
        "precision_at_conf_70": 0.75,
        "per_symbol": {
            "BTCUSDT": {"net_pnl_usd": 1500.0, "trades": 10},
            "ETHUSDT": {"net_pnl_usd": 1500.0, "trades": 10},
        },
        "ev_by_confidence_bucket": {"0.70-0.80": 50.0},
        "internal_model_signals": {"LONG": 10, "SHORT": 10, "SKIP": 0},
        "publishable_signals_ge_70": {"LONG": 10, "SHORT": 10},
        "backtest_publishable_ge_70": {"trade_count": 20, "net_pnl_usd": 3000.0},
        "publishable_long_count_ge_70": 10,
        "publishable_short_count_ge_70": 10,
        "publishable_long_ev_ge_70": 1500.0,
        "publishable_short_ev_ge_70": 1500.0,
        "publishable_total_ev_ge_70": 3000.0,
        "publishable_total_trade_count_ge_70": 20,
    }
    result = evaluate_promotion(metrics, rules=PromotionRules(min_trades=20))
    assert result.passed is True


def test_build_evaluation_report_separates_internal_and_publishable_pnl() -> None:
    batch = _batch(
        ["LONG", "SHORT", "SHORT"],
        [0.8, 0.4, 0.75],
        [True, False, True],
    )
    report = build_evaluation_report(batch, publish_confidence=0.70)
    internal = report["backtest_internal_all_signals"]
    publishable = report["backtest_publishable_confidence_ge_0.70"]
    assert internal["trade_count"] == 3
    assert publishable["trade_count"] == 2
    assert internal["net_pnl_usd"] == 300.0
    assert publishable["net_pnl_usd"] == 200.0


def test_promote_artifacts_removes_stale_files(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_run"
    production = tmp_path / "artifacts"
    candidate.mkdir()
    production.mkdir()

    (candidate / "tabular.joblib").write_text("candidate-tabular")
    (candidate / "summary.json").write_text(json.dumps({"run": "candidate"}))
    (production / "tabular.joblib").write_text("old-tabular")
    (production / "meta_model.joblib").write_text("stale-legacy-meta")

    promote_artifacts(candidate, production, backup=False)

    assert not (production / "meta_model.joblib").exists()
    assert (production / "tabular.joblib").read_text() == "candidate-tabular"
    assert (production / "summary.json").read_text() == (candidate / "summary.json").read_text()
    verify_artifacts_match(candidate, production)


def test_promote_artifacts_creates_backup(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_run"
    production = tmp_path / "artifacts"
    candidate.mkdir()
    production.mkdir()

    (candidate / "model.txt").write_text("v2")
    (production / "model.txt").write_text("v1")
    (production / "meta_model.joblib").write_text("stale")

    backup_path = promote_artifacts(candidate, production, backup=True)

    assert backup_path is not None
    assert backup_path.is_dir()
    assert (backup_path / "meta_model.joblib").read_text() == "stale"
    assert not (production / "meta_model.joblib").exists()
    verify_artifacts_match(candidate, production)
