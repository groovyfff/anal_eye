"""Smoke tests for the pure-numerical core (no heavy ML deps required).

These validate the parts of the pipeline that must be correct regardless of
whether torch / lightgbm / sb3 are installed: the EV gate math, ATR/Kelly
sizing, feature engineering shape, and the fusion decision logic + JSON shape.
"""

from __future__ import annotations

import numpy as np

from ae_brain.config import get_settings
from ae_brain.contracts import (
    Decision,
    FinalSignal,
    LayerProbabilities,
    Side,
    TradeCandidate,
)
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.features.schema import FEATURE_NAMES, n_features
from ae_brain.layers.fusion import FusionContext, FusionLayer
from ae_brain.layers.meta import CLASS_LONG, CLASS_SHORT, resolve_directional_class
from ae_brain.risk.costs import CostModel
from ae_brain.risk.ev_gate import EVGate
from ae_brain.risk.sizing import PositionSizer
from ae_brain.training.synthetic import generate_synthetic_candles


def _components():
    s = get_settings()
    cost = CostModel(s.cost)
    gate = EVGate(cost, min_ev_usd=s.fusion.min_ev_usd)
    sizer = PositionSizer(s.risk)
    fusion = FusionLayer(s.fusion, s.risk, gate, sizer)
    return s, cost, gate, sizer, fusion


def test_feature_schema_size():
    assert 55 <= n_features() <= 70
    assert len(set(FEATURE_NAMES)) == n_features()  # no duplicates


def test_feature_engineering_shape():
    candles = generate_synthetic_candles(n=600, seed=1)
    eng = FeatureEngineer()
    frame = eng.compute_frame(candles)
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == len(candles)
    assert np.isfinite(frame.to_numpy()).all()  # no NaN/inf leaks


def test_ev_gate_formula_exact():
    _, cost, gate, _, _ = _components()
    # Construct a clearly positive-EV long.
    res = gate.evaluate(
        side=Side.LONG,
        entry=100.0,
        take_profit=105.0,
        stop_loss=98.0,
        notional_usd=10_000.0,
        prob_tp=0.6,
        prob_sl=0.3,
    )
    # Recompute the mandated formula independently.
    expected = (res.prob_tp * res.net_reward) - (res.prob_sl * res.net_risk)
    assert abs(res.expected_value - round(expected, 6)) < 1e-6
    assert res.is_positive_ev is (res.expected_value > 0)


def test_ev_gate_rejects_negative_edge():
    _, cost, gate, _, _ = _components()
    res = gate.evaluate(
        side=Side.LONG,
        entry=100.0,
        take_profit=100.5,
        stop_loss=95.0,
        notional_usd=10_000.0,
        prob_tp=0.5,
        prob_sl=0.5,
    )
    assert res.is_positive_ev is False


def test_sizing_no_hardcoded_stop_and_caps():
    s, _, _, sizer, _ = _components()
    r = sizer.size(
        entry=100.0, atr=2.0, side=Side.LONG, prob_tp=0.62, reward_risk_ratio=1.67
    )
    # ATR-based stop, not a fixed -5%.
    assert abs(r.stop_distance - 2.0 * s.risk.atr_sl_mult) < 1e-9
    assert 0.0 <= r.position_size_pct <= s.risk.max_position_pct
    assert 0.0 <= r.leverage <= s.risk.max_leverage


def test_correlation_limit_blocks():
    s, _, _, sizer, _ = _components()
    r = sizer.size(
        entry=100.0, atr=2.0, side=Side.LONG, prob_tp=0.7, reward_risk_ratio=1.67,
        correlated_exposure=s.risk.max_correlated_exposure + 0.1,
    )
    assert r.position_size_pct == 0.0
    assert r.rejected_reason == "correlation_budget_exhausted"


def test_meta_threshold_directional_resolution():
    """Directional class ignores p_skip; threshold + argmax tie-break on p_long/p_short."""
    assert resolve_directional_class(0.299, 0.336, threshold=0.30)[0] == CLASS_LONG
    assert resolve_directional_class(0.25, 0.28, threshold=0.30)[0] is None
    assert resolve_directional_class(0.35, 0.40, threshold=0.30)[0] == CLASS_LONG
    assert resolve_directional_class(0.40, 0.35, threshold=0.30)[0] == CLASS_SHORT
    assert resolve_directional_class(0.36, 0.37, threshold=0.30, margin=0.05)[0] is None


def test_fusion_skip_on_low_conviction():
    _, _, _, _, fusion = _components()
    probs = LayerProbabilities(
        tabular_p_up=0.51, sequence_p_continuation=0.5, sequence_trend_sign=0.0,
        rl_target_exposure=0.0,
    )
    ctx = FusionContext(symbol="BTCUSDT", entry_price=100.0, atr=2.0)
    sig = fusion.decide(probs, ctx)
    assert sig.decision == Decision.SKIP


def test_trade_candidate_contract_from_message():
    candles = generate_synthetic_candles(n=60, seed=3)
    candles["ts"] = candles["ts"].astype(str)
    payload = {
        "symbol": "AAPL",
        "interval": "5m",
        "signal_log_db_id": 4242,
        "asset_class": "stock",
        "candles": candles.to_dict(orient="records"),
    }
    cand = TradeCandidate.from_message(payload)
    assert cand.signal_log_db_id == 4242
    assert cand.asset_class == "stock"
    assert cand.is_derivative is False
    assert len(cand.candles) >= 48  # sequence-window requirement


def test_trade_candidate_missing_signal_log_db_id_defaults_to_zero():
    # Legacy crypto producers omit the id -> default 0 -> INSERT fallback path.
    cand = TradeCandidate.from_message({"symbol": "X", "candles": []})
    assert cand.signal_log_db_id == 0
    assert cand.asset_class == "crypto"


def test_trade_candidate_null_signal_log_db_id_defaults_to_zero():
    cand = TradeCandidate.from_message(
        {"symbol": "X", "candles": [], "signal_log_db_id": None}
    )
    assert cand.signal_log_db_id == 0


def test_trade_candidate_invalid_asset_class_defaults_to_crypto():
    cand = TradeCandidate(symbol="X", interval="5m", candles=[], signal_log_db_id=1,
                          asset_class="banana")
    assert cand.asset_class == "crypto"
    assert cand.is_derivative is True


def test_feature_engineering_null_microstructure_non_crypto():
    """Stocks/forex/metals send null funding/OI/CVD/liquidations -> no crash."""
    candles = generate_synthetic_candles(n=400, seed=5)
    # Simulate a traditional-asset payload: derivatives fields all null.
    for col in ("funding_rate", "open_interest", "taker_buy_volume",
                "long_liq_notional", "short_liq_notional", "basis"):
        candles[col] = None
    eng = FeatureEngineer()
    frame = eng.compute_frame(candles)
    assert list(frame.columns) == list(FEATURE_NAMES)
    arr = frame.to_numpy()
    assert np.isfinite(arr).all()  # neutral fallbacks, no NaN/inf leaks
    # Derivative-only features collapse to their neutral value (0.0).
    assert abs(float(frame["funding_rate"].iloc[-1])) < 1e-9
    assert abs(float(frame["oi_z"].iloc[-1])) < 1e-9


def test_engineer_latest_null_funding_is_safe():
    from ae_brain.inference.engine import _engineer_latest

    candles = generate_synthetic_candles(n=60, seed=9)
    candles["funding_rate"] = None  # null funding (e.g. a stock)
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")
    ctx = _engineer_latest(rows, z_window=50, asset_class="stock")
    assert ctx["funding_rate"] == 0.0
    assert ctx["entry_price"] > 0
    assert np.isfinite(ctx["features"]).all()


def test_db_signal_columns_flatten():
    """The shared UPDATE/INSERT column map carries all ensemble outputs."""
    from ae_brain.data.database import Database

    _, _, _, _, fusion = _components()
    probs = LayerProbabilities(tabular_p_up=0.7, sequence_p_continuation=0.65,
                               sequence_trend_sign=1.0, rl_target_exposure=0.5)
    ctx = FusionContext(symbol="BTCUSDT", entry_price=100.0, atr=1.0, adv_usd=5_000_000)
    sig = fusion.decide(probs, ctx)
    cols = Database._signal_columns({"vol_z": 0.1}, probs.as_dict(), sig, "crypto")
    for key in ("asset_class", "decision", "expected_value_usd", "kelly_fraction",
                "metrics", "tabular_p_up", "is_positive_ev"):
        assert key in cols
    assert cols["asset_class"] == "crypto"
    assert cols["decision"] in {"LONG", "SHORT", "SKIP"}


def test_fusion_long_on_strong_positive_ev():
    _, _, _, _, fusion = _components()
    probs = LayerProbabilities(
        tabular_p_up=0.78, sequence_p_continuation=0.8, sequence_trend_sign=1.0,
        rl_target_exposure=0.7,
    )
    ctx = FusionContext(symbol="BTCUSDT", entry_price=100.0, atr=1.0, adv_usd=5_000_000)
    sig = fusion.decide(probs, ctx)
    d = sig.to_dict()
    assert set(["decision", "position_size_pct", "leverage", "take_profit", "stop_loss"]).issubset(d)
    assert d["decision"] in {"LONG", "SHORT", "SKIP"}
    if d["decision"] == "LONG":
        assert d["take_profit"] > ctx.entry_price > d["stop_loss"]
