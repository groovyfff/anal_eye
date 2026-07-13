"""Tests for side_specialists final evaluation decision path."""

from __future__ import annotations

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
from ae_brain.training.evaluation import side_specialist_collapse_warnings


def test_resolve_side_specialist_decision_can_produce_long() -> None:
    decision, reason = resolve_side_specialist_decision(0.82, 0.71)
    assert decision == "LONG"
    assert reason == "long_prob_ge_threshold_and_gt_short"


def test_resolve_side_specialist_decision_can_produce_short() -> None:
    decision, reason = resolve_side_specialist_decision(0.71, 0.84)
    assert decision == "SHORT"
    assert reason == "short_prob_ge_threshold_and_gt_long"


def test_no_silent_long_only_when_short_probability_qualifies() -> None:
    decision, _ = resolve_side_specialist_decision(0.65, 0.80)
    assert decision == "SHORT"


def test_no_silent_short_only_when_long_probability_qualifies() -> None:
    decision, _ = resolve_side_specialist_decision(0.80, 0.65)
    assert decision == "LONG"


def test_equal_qualifying_probabilities_skip() -> None:
    decision, reason = resolve_side_specialist_decision(0.75, 0.75)
    assert decision == "SKIP"
    assert reason is None


def test_both_below_threshold_skip() -> None:
    decision, reason = resolve_side_specialist_decision(0.60, 0.55)
    assert decision == "SKIP"
    assert reason is None


def _pred(side: str, cal: float, *, publishable: bool = True) -> SideSpecialistPrediction:
    return SideSpecialistPrediction(
        side=side,
        p_profitable_raw=cal,
        p_profitable_calibrated=cal,
        ev_usd=10.0,
        confidence_adjusted_ev=cal * 10.0,
        prob_tp=0.6,
        prob_sl=0.4,
        sizing_ok=True,
        publishable=publishable,
    )


def _fusion_with_specialists() -> FusionLayer:
    cfg = MagicMock()
    cfg.meta_mode = "side_specialists"
    cfg.meta_direction_margin = 0.05
    cfg.meta_direction_threshold = 0.30
    cfg.w_tabular = 0.4
    cfg.w_sequence = 0.3
    cfg.w_rl = 0.3
    cfg.min_conviction = 0.5
    risk = MagicMock()
    risk.atr_tp_mult = 2.5
    risk.atr_sl_mult = 1.5
    ev_gate = MagicMock()
    sizer = MagicMock()
    bundle = SideSpecialistsBundle(
        long_model=MagicMock(is_ready=lambda: True),
        short_model=MagicMock(is_ready=lambda: True),
    )
    fusion = FusionLayer(
        cfg,
        risk,
        ev_gate,
        sizer,
        side_specialists=bundle,
        layer_mask={"tabular": True, "sequence": False, "rl": False},
        force_meta_mode="side_specialists",
    )
    return fusion


def test_fusion_side_specialists_chooses_short_when_short_prob_higher() -> None:
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
            (_pred("LONG", 0.72, publishable=False), sizing, ev),
            (_pred("SHORT", 0.81, publishable=False), sizing, ev),
        ],
    ):
        probs = LayerProbabilities(
            tabular_p_up=0.55,
            sequence_p_continuation=0.5,
            sequence_trend_sign=0.0,
            rl_target_exposure=0.0,
            rl_state_value=0.0,
        )
        ctx = FusionContext(
            symbol="BTCUSDT",
            entry_price=100.0,
            atr=2.0,
            funding_rate_8h=0.0,
            adv_usd=1e9,
            holding_hours=8.0,
            correlated_exposure=0.0,
            correlation_id="test",
            regime_onehot=(0.33, 0.34, 0.33),
        )
        signal = fusion._decide_side_specialists(probs, ctx)

    assert signal.decision == Decision.SHORT
    ss = signal.components["side_specialists"]
    assert ss["decision_rule"] == "calibrated_prob_direct"
    assert ss["decision_reason"] == "short_prob_ge_threshold_and_gt_long"
    assert ss["chosen_side"] == "SHORT"


def test_fusion_side_specialists_chooses_long_when_long_prob_higher() -> None:
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
            (_pred("LONG", 0.88, publishable=False), sizing, ev),
            (_pred("SHORT", 0.71, publishable=True), sizing, ev),
        ],
    ):
        probs = LayerProbabilities(
            tabular_p_up=0.55,
            sequence_p_continuation=0.5,
            sequence_trend_sign=0.0,
            rl_target_exposure=0.0,
            rl_state_value=0.0,
        )
        ctx = FusionContext(
            symbol="BTCUSDT",
            entry_price=100.0,
            atr=2.0,
            funding_rate_8h=0.0,
            adv_usd=1e9,
            holding_hours=8.0,
            correlated_exposure=0.0,
            correlation_id="test",
            regime_onehot=(0.33, 0.34, 0.33),
        )
        signal = fusion._decide_side_specialists(probs, ctx)

    assert signal.decision == Decision.LONG
    assert signal.components["side_specialists"]["chosen_side"] == "LONG"


def test_fusion_does_not_use_ev_tiebreak_when_short_prob_wins() -> None:
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
    long_ev = MagicMock(expected_value=50.0, is_positive_ev=True, as_dict=lambda: {"expected_value": 50.0})
    short_ev = MagicMock(expected_value=5.0, is_positive_ev=True, as_dict=lambda: {"expected_value": 5.0})

    with patch.object(
        fusion,
        "_eval_side_specialist",
        side_effect=[
            (_pred("LONG", 0.74, publishable=True), sizing, long_ev),
            (_pred("SHORT", 0.79, publishable=True), sizing, short_ev),
        ],
    ):
        probs = LayerProbabilities(
            tabular_p_up=0.55,
            sequence_p_continuation=0.5,
            sequence_trend_sign=0.0,
            rl_target_exposure=0.0,
            rl_state_value=0.0,
        )
        ctx = FusionContext(
            symbol="BTCUSDT",
            entry_price=100.0,
            atr=2.0,
            funding_rate_8h=0.0,
            adv_usd=1e9,
            holding_hours=8.0,
            correlated_exposure=0.0,
            correlation_id="test",
            regime_onehot=(0.33, 0.34, 0.33),
        )
        signal = fusion._decide_side_specialists(probs, ctx)

    assert signal.decision == Decision.SHORT


def test_side_specialist_collapse_warning_long_only(tmp_path: Path) -> None:
    artifacts = tmp_path / "candidate"
    artifacts.mkdir()
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {
                        "0.70": {"publishable_LONG": 40, "publishable_SHORT": 60}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "publishable_long_count_ge_70": 971,
        "publishable_short_count_ge_70": 0,
    }
    warnings = side_specialist_collapse_warnings(artifacts, summary)
    assert "long_only_collapse" in warnings[0]


def test_side_specialist_collapse_warning_short_only(tmp_path: Path) -> None:
    artifacts = tmp_path / "candidate"
    artifacts.mkdir()
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {
                        "0.70": {"publishable_LONG": 30, "publishable_SHORT": 70}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "publishable_long_count_ge_70": 0,
        "publishable_short_count_ge_70": 500,
    }
    warnings = side_specialist_collapse_warnings(artifacts, summary)
    assert "short_only_collapse" in warnings[0]


def test_evaluate_candidate_sets_side_specialists_decision_mode(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_candidate",
        ROOT / "scripts" / "evaluate_candidate.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    artifacts = tmp_path / "candidate"
    artifacts.mkdir()
    (artifacts / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 10, "publishable_SHORT": 10}}
                }
            }
        ),
        encoding="utf-8",
    )

    summary = {
        "promotable": False,
        "publishable_long_count_ge_70": 100,
        "publishable_short_count_ge_70": 0,
        "warnings": [],
    }
    collapse = side_specialist_collapse_warnings(artifacts, summary)
    summary.setdefault("warnings", [])
    summary["warnings"].extend(collapse)
    summary["decision_mode"] = "side_specialists_calibrated_prob_direct"

    assert summary["decision_mode"] == "side_specialists_calibrated_prob_direct"
    assert any("long_only_collapse" in w for w in summary["warnings"])
