"""Tests for side_specialists final evaluation decision path."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_brain.contracts import Decision
from ae_brain.layers.fusion import FusionContext, FusionLayer, LayerProbabilities
from ae_brain.layers.side_specialists import (
    SideSpecialistPrediction,
    SideSpecialistsBundle,
    resolve_side_specialist_decision,
)
from ae_brain.training.evaluation import SignalBatch, build_evaluation_report, side_specialist_collapse_warnings
from ae_brain.training.side_specialist_decisions import (
    decision_from_calibrated_probs,
    rebuild_decision_from_components,
)


def test_regression_long_0_82_short_0_71() -> None:
    decision, reason = decision_from_calibrated_probs(0.82, 0.71)
    assert decision == "LONG"
    assert reason in {"long_prob_ge_threshold", "long_prob_ge_threshold_and_gt_short", "both_qualify_ev_tiebreak_long"}


def test_regression_long_0_72_short_0_81() -> None:
    decision, reason = decision_from_calibrated_probs(0.72, 0.81)
    assert decision == "SHORT"
    assert reason in {"short_prob_ge_threshold", "short_prob_ge_threshold_and_gt_long", "both_qualify_ev_tiebreak_short"}


def test_regression_long_0_69_short_0_69() -> None:
    decision, reason = decision_from_calibrated_probs(0.69, 0.69)
    assert decision == "SKIP"
    assert reason is None


def test_negative_short_ev_does_not_force_long_when_short_prob_higher() -> None:
    """CostModel EV bias must not wipe SHORT when short_prob wins."""
    decision, reason = decision_from_calibrated_probs(
        0.72, 0.81, long_ev=50.0, short_ev=-5.0
    )
    assert decision == "SHORT"
    assert reason == "short_prob_ge_threshold_and_gt_long"


def test_rebuild_from_components_short() -> None:
    components = {
        "side_specialists": {
            "long": {"p_profitable_calibrated": 0.72, "ev_usd": 10.0},
            "short": {"p_profitable_calibrated": 0.81, "ev_usd": 12.0},
        }
    }
    decision, reason, lp, sp = rebuild_decision_from_components(components)
    assert decision == "SHORT"
    assert lp == pytest.approx(0.72)
    assert sp == pytest.approx(0.81)
    assert reason is not None


def test_both_qualify_ev_tiebreak_prefers_long() -> None:
    decision, reason = resolve_side_specialist_decision(
        0.74, 0.79, long_ev=50.0, short_ev=5.0
    )
    assert decision == "LONG"
    assert reason == "both_qualify_ev_tiebreak_long"


def test_both_qualify_ev_tiebreak_prefers_short() -> None:
    decision, reason = resolve_side_specialist_decision(
        0.79, 0.74, long_ev=5.0, short_ev=50.0
    )
    assert decision == "SHORT"
    assert reason == "both_qualify_ev_tiebreak_short"


def test_both_qualify_equal_ev_uses_probability() -> None:
    decision, reason = resolve_side_specialist_decision(
        0.72, 0.81, long_ev=10.0, short_ev=10.0
    )
    assert decision == "SHORT"
    assert reason == "short_prob_ge_threshold_and_gt_long"


def test_only_long_qualifies() -> None:
    decision, reason = resolve_side_specialist_decision(0.80, 0.65)
    assert decision == "LONG"
    assert reason == "long_prob_ge_threshold"


def test_only_short_qualifies() -> None:
    decision, reason = resolve_side_specialist_decision(0.65, 0.80)
    assert decision == "SHORT"
    assert reason == "short_prob_ge_threshold"


def test_both_equal_qualifying_probabilities_skip() -> None:
    decision, reason = resolve_side_specialist_decision(0.75, 0.75)
    assert decision == "SKIP"
    assert reason == "both_qualify_equal_prob_skip"


def _pred(side: str, cal: float, *, ev: float = 10.0) -> SideSpecialistPrediction:
    return SideSpecialistPrediction(
        side=side,
        p_profitable_raw=cal,
        p_profitable_calibrated=cal,
        ev_usd=ev,
        confidence_adjusted_ev=cal * ev,
        prob_tp=0.6,
        prob_sl=0.4,
        sizing_ok=True,
        publishable=True,
    )


def _fusion_with_specialists() -> FusionLayer:
    cfg = MagicMock()
    cfg.meta_mode = "side_specialists"
    risk = MagicMock()
    risk.atr_tp_mult = 2.5
    risk.atr_sl_mult = 1.5
    bundle = SideSpecialistsBundle(
        long_model=MagicMock(is_ready=lambda: True),
        short_model=MagicMock(is_ready=lambda: True),
    )
    return FusionLayer(
        cfg,
        risk,
        MagicMock(),
        MagicMock(),
        side_specialists=bundle,
        layer_mask={"tabular": True, "sequence": False, "rl": False},
        force_meta_mode="side_specialists",
    )


def test_fusion_uses_0_70_threshold_and_short_prob_wins() -> None:
    fusion = _fusion_with_specialists()
    sizing = MagicMock(
        position_size_pct=0.02,
        leverage=2.0,
        take_profit=110.0,
        stop_loss=90.0,
        rejected_reason=None,
        kelly_fraction_raw=0.1,
        correlation_scale=1.0,
        notional_usd=100.0,
    )
    ev = MagicMock(expected_value=12.0, is_positive_ev=True, as_dict=lambda: {"expected_value": 12.0})

    with patch.object(
        fusion,
        "_eval_side_specialist",
        side_effect=[
            (_pred("LONG", 0.72, ev=12.0), sizing, ev),
            (_pred("SHORT", 0.81, ev=12.0), sizing, ev),
        ],
    ):
        signal = fusion._decide_side_specialists(
            LayerProbabilities(0.55, 0.5, 0.0, 0.0, 0.0),
            FusionContext(
                symbol="BTCUSDT",
                entry_price=100.0,
                atr=2.0,
                funding_rate_8h=0.0,
                adv_usd=1e9,
                holding_hours=8.0,
                correlated_exposure=0.0,
                correlation_id="test",
                regime_onehot=(0.33, 0.34, 0.33),
            ),
        )

    assert signal.decision == Decision.SHORT
    assert signal.components["side_specialists"]["decision_rule"] == "calibrated_prob_direct"


def test_fusion_does_not_skip_when_both_sides_qualify() -> None:
    fusion = _fusion_with_specialists()
    sizing = MagicMock(
        position_size_pct=0.02,
        leverage=2.0,
        take_profit=110.0,
        stop_loss=90.0,
        rejected_reason=None,
        kelly_fraction_raw=0.1,
        correlation_scale=1.0,
        notional_usd=100.0,
    )
    ev = MagicMock(expected_value=12.0, is_positive_ev=True, as_dict=lambda: {"expected_value": 12.0})

    with patch.object(
        fusion,
        "_eval_side_specialist",
        side_effect=[
            (_pred("LONG", 0.82, ev=12.0), sizing, ev),
            (_pred("SHORT", 0.71, ev=12.0), sizing, ev),
        ],
    ):
        signal = fusion._decide_side_specialists(
            LayerProbabilities(0.55, 0.5, 0.0, 0.0, 0.0),
            FusionContext(
                symbol="BTCUSDT",
                entry_price=100.0,
                atr=2.0,
                funding_rate_8h=0.0,
                adv_usd=1e9,
                holding_hours=8.0,
                correlated_exposure=0.0,
                correlation_id="test",
                regime_onehot=(0.33, 0.34, 0.33),
            ),
        )

    assert signal.decision == Decision.LONG
    assert signal.components["side_specialists"]["decision_reason"] != "both_qualify_ambiguous_skip"


def test_fake_probability_arrays_feed_final_report_short_and_long() -> None:
    """Decision vector from calibrated probs must drive publishable report counts."""
    rows = [
        decision_from_calibrated_probs(0.72, 0.81),  # SHORT
        decision_from_calibrated_probs(0.82, 0.71),  # LONG
        decision_from_calibrated_probs(0.69, 0.69),  # SKIP
        decision_from_calibrated_probs(0.65, 0.80),  # SHORT
        decision_from_calibrated_probs(0.90, 0.60),  # LONG
    ]
    decisions = np.array([d for d, _ in rows])
    conf = np.array([0.81, 0.82, 0.0, 0.80, 0.90])
    publishable = np.array([d in ("LONG", "SHORT") for d in decisions])
    batch = SignalBatch(
        decisions=decisions,
        expected_evs=np.where(publishable, 5.0, 0.0),
        confidence=conf,
        symbols=np.array(["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        timestamps=np.array(["2024-01-01T00:00:00Z"] * 5),
        publishable=publishable,
    )
    report = build_evaluation_report(batch, publish_confidence=0.70)
    assert report["publishable_signals_confidence_ge_0.70"]["SHORT"] == 2
    assert report["publishable_signals_confidence_ge_0.70"]["LONG"] == 2
    assert report["internal_model_signals"]["SKIP"] == 1
    assert report["publishable_short_count_ge_70"] == 2


def test_balanced_backtest_report_has_both_publishable_sides() -> None:
    n = 200
    decisions = np.array(["LONG"] * 100 + ["SHORT"] * 100)
    publishable = np.ones(n, dtype=bool)
    batch = SignalBatch(
        decisions=decisions,
        expected_evs=np.full(n, 5.0),
        confidence=np.full(n, 0.75),
        symbols=np.array(["BTCUSDT"] * n),
        timestamps=np.array(["2024-01-01T00:00:00Z"] * n),
        publishable=publishable,
    )
    report = build_evaluation_report(batch, publish_confidence=0.70)
    pub = report["publishable_signals_confidence_ge_0.70"]
    assert pub["LONG"] == 100
    assert pub["SHORT"] == 100
    assert report["publishable_short_count_ge_70"] == 100
    assert report["publishable_long_count_ge_70"] == 100


def test_collapse_warning_triggers_on_long_only_summary(tmp_path: Path) -> None:
    artifacts = tmp_path / "candidate"
    artifacts.mkdir()
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 40, "publishable_SHORT": 60}}
                }
            }
        ),
        encoding="utf-8",
    )
    summary = {"publishable_long_count_ge_70": 971, "publishable_short_count_ge_70": 0}
    warnings = side_specialist_collapse_warnings(artifacts, summary)
    assert any("long_only_collapse" in w for w in warnings)


def test_no_collapse_warning_when_both_sides_present(tmp_path: Path) -> None:
    artifacts = tmp_path / "candidate"
    artifacts.mkdir()
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 40, "publishable_SHORT": 60}}
                }
            }
        ),
        encoding="utf-8",
    )
    summary = {"publishable_long_count_ge_70": 40, "publishable_short_count_ge_70": 60}
    assert side_specialist_collapse_warnings(artifacts, summary) == []


def _load_evaluate_candidate():
    spec = importlib.util.spec_from_file_location(
        "evaluate_candidate",
        ROOT / "scripts" / "evaluate_candidate.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_candidate_integration_side_specialists_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate = _load_evaluate_candidate()
    artifacts = tmp_path / "top200_test_side_spec"
    artifacts.mkdir()
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"placeholder")

    (artifacts / "training_summary.json").write_text(
        json.dumps({"meta": {"meta_mode": "side_specialists"}}),
        encoding="utf-8",
    )
    (artifacts / "side_configs.json").write_text(json.dumps({"long": {}, "short": {}}))
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 50, "publishable_SHORT": 50}}
                }
            }
        ),
        encoding="utf-8",
    )

    n = 120
    batch = SignalBatch(
        decisions=np.array(["LONG"] * 60 + ["SHORT"] * 60),
        expected_evs=np.full(n, 8.0),
        confidence=np.full(n, 0.78),
        symbols=np.array(["BTCUSDT"] * 30 + ["ETHUSDT"] * 30 + ["BTCUSDT"] * 30 + ["ETHUSDT"] * 30),
        timestamps=np.array(["2024-06-01T00:00:00Z"] * n),
        publishable=np.ones(n, dtype=bool),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_candidate.py",
            "--run-id",
            artifacts.name,
            "--candidates-dir",
            str(tmp_path),
            "--dataset",
            str(dataset),
            "--symbols",
            "BTCUSDT",
            "--report-dir",
            str(tmp_path / "reports"),
        ],
    )

    with patch.object(evaluate, "asyncio") as mock_asyncio:
        mock_asyncio.run.side_effect = [(batch, MagicMock()), {}]
        with patch.object(evaluate, "_training_label_audit", return_value={}):
            with patch.object(evaluate, "_meta_model_audit", return_value={}):
                with patch.object(evaluate, "_test_vol_z_series", return_value=None):
                        with patch.object(evaluate, "_regime_filter_evaluation", return_value={"applied": False}):
                            with patch.object(evaluate, "_diagnose_batch", return_value={}):
                                evaluate.main()

    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision_mode"] == "side_specialists_calibrated_prob_direct"
    assert summary["publishable_signals_ge_70"]["LONG"] == 60
    assert summary["publishable_signals_ge_70"]["SHORT"] == 60
    assert summary.get("side_specialist_collapse_detected") is not True
    assert not side_specialist_collapse_warnings(artifacts, summary)
    pub_report = json.loads((artifacts / "reports" / "publishable_report.json").read_text(encoding="utf-8"))
    assert pub_report["publishable_short_count_ge_70"] == 60
    assert pub_report["publishable_long_count_ge_70"] == 60
    assert pub_report["decision_mode"] == "side_specialists_calibrated_prob_direct"


def test_evaluate_candidate_hard_fails_on_collapse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate = _load_evaluate_candidate()
    artifacts = tmp_path / "top200_collapse"
    artifacts.mkdir()
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"placeholder")

    (artifacts / "training_summary.json").write_text(
        json.dumps({"meta": {"meta_mode": "side_specialists"}}),
        encoding="utf-8",
    )
    (artifacts / "side_configs.json").write_text(json.dumps({"long": {}, "short": {}}))
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 50, "publishable_SHORT": 50}}
                }
            }
        ),
        encoding="utf-8",
    )

    n = 120
    batch = SignalBatch(
        decisions=np.array(["LONG"] * 118 + ["SHORT"] * 2),
        expected_evs=np.full(n, 8.0),
        confidence=np.concatenate([np.full(118, 0.78), np.full(2, 0.78)]),
        symbols=np.array(["BTCUSDT"] * n),
        timestamps=np.array(["2024-06-01T00:00:00Z"] * n),
        publishable=np.array([True] * 118 + [False] * 2),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_candidate.py",
            "--run-id",
            artifacts.name,
            "--candidates-dir",
            str(tmp_path),
            "--dataset",
            str(dataset),
            "--symbols",
            "BTCUSDT",
            "--report-dir",
            str(tmp_path / "reports"),
        ],
    )

    with patch.object(evaluate, "asyncio") as mock_asyncio:
        mock_asyncio.run.side_effect = [(batch, MagicMock()), {}]
        with patch.object(evaluate, "_training_label_audit", return_value={}):
            with patch.object(evaluate, "_meta_model_audit", return_value={}):
                with patch.object(evaluate, "_test_vol_z_series", return_value=None):
                    with patch.object(evaluate, "_regime_filter_evaluation", return_value={"applied": False}):
                        with patch.object(evaluate, "_diagnose_batch", return_value={}):
                            with pytest.raises(SystemExit) as exc:
                                evaluate.main()
    assert exc.value.code == 3
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("side_specialist_collapse_detected") is True
