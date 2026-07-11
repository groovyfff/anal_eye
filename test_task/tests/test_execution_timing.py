"""Lookahead-bias tests for execution timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ae_brain.contracts import Side, TradeCandidate
from ae_brain.execution.timing import (
    candles_for_features,
    compare_entry_pnl_delta,
    resolve_execution_timing,
    reprice_tp_sl_for_entry,
)
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.inference.engine import _engineer_latest
from ae_brain.messaging.candidate_normalizer import normalize_candidate
from ae_brain.risk.sizing import PositionSizer
from ae_brain.config import RiskConfig


def _candle_rows(n: int, *, start: float = 100.0) -> list[dict]:
    rows: list[dict] = []
    base_ms = 1_700_000_000_000
    for i in range(n):
        o = start + i
        rows.append(
            {
                "timestamp": base_ms + i * 3_600_000,
                "open": o,
                "high": o + 2,
                "low": o - 1,
                "close": o + 0.5,
                "volume": 1000.0 + i,
                "closed": True,
                "close_time": base_ms + i * 3_600_000 + 3_599_999,
            }
        )
    return rows


def test_signal_from_candle_t_enters_on_t_plus_one_open() -> None:
    candles = _candle_rows(5)
    next_open = 106.0
    timing = resolve_execution_timing(
        candles,
        {
            "next_candle_open": next_open,
            "next_candle_open_time": "2026-01-02T01:00:00Z",
        },
    )
    assert timing.signal_reference_price == pytest.approx(candles[-1]["close"])
    assert timing.execution_price == pytest.approx(next_open)
    assert timing.execution_price_source == "next_open"
    assert timing.execution_time == "2026-01-02T01:00:00Z"


def test_features_use_only_through_signal_candle_t() -> None:
    candles = _candle_rows(60)
    eng = FeatureEngineer(z_window=20)
    df1 = pd.DataFrame(candles_for_features(candles))
    df2 = df1.copy()
    df2.loc[df2.index[-1], "close"] = float(df2.loc[df2.index[-1], "close"]) * 2.0
    frame1 = eng.compute_frame(df1)
    frame2 = eng.compute_frame(df2)
    assert float(frame1.iloc[-2]["ret_1"]) == pytest.approx(float(frame2.iloc[-2]["ret_1"]))
    assert float(frame1.iloc[-1]["ret_1"]) != float(frame2.iloc[-1]["ret_1"])


def test_backtest_pnl_changes_when_next_open_differs_from_t_close() -> None:
    signal_close = 100.0
    next_open = 101.5
    sizer = PositionSizer(RiskConfig())
    sizing_close = sizer.size(entry=signal_close, atr=2.0, side=Side.LONG, prob_tp=0.6, reward_risk_ratio=2.0)
    sizing_open = sizer.size(entry=next_open, atr=2.0, side=Side.LONG, prob_tp=0.6, reward_risk_ratio=2.0)
    assert sizing_close.take_profit != sizing_open.take_profit
    assert sizing_close.stop_loss != sizing_open.stop_loss
    delta = compare_entry_pnl_delta(
        side=Side.LONG,
        signal_close_entry=signal_close,
        execution_entry=next_open,
        take_profit=sizing_open.take_profit,
        stop_loss=sizing_open.stop_loss,
        notional_usd=sizing_open.notional_usd,
    )
    assert delta != 0.0


def test_reprice_tp_sl_preserves_distance_from_entry() -> None:
    old_entry = 100.0
    new_entry = 101.0
    tp, sl = reprice_tp_sl_for_entry(
        side=Side.LONG,
        old_entry=old_entry,
        new_entry=new_entry,
        take_profit=110.0,
        stop_loss=95.0,
    )
    assert tp - new_entry == pytest.approx(10.0)
    assert new_entry - sl == pytest.approx(5.0)


def test_live_candidate_execution_not_before_close_time() -> None:
    candles = _candle_rows(3)
    timing = resolve_execution_timing(candles, {"mark_price": candles[-1]["close"]})
    from ae_brain.execution.timing import _to_ms

    assert timing.execution_price_source == "live_mark_after_close"
    assert _to_ms(timing.execution_time) >= _to_ms(timing.signal_candle_close_time)
    assert timing.execution_price >= timing.signal_reference_price


def test_binance_payload_normalizes_with_execution_fields() -> None:
    candles = _candle_rows(200)
    payload = {
        "symbol": "BTCUSDT",
        "asset_class": "crypto",
        "timeframe": "1h",
        "current_price": candles[-1]["close"],
        "composite_score": 0.8,
        "candle_open_time": "2026-01-01T00:00:00Z",
        "candle_close_time": "2026-01-01T00:59:59Z",
        "candles": candles,
        "features": {"current_price": candles[-1]["close"], "funding_rate": 0.0},
    }
    norm = normalize_candidate(payload)
    assert norm.skip_reason is None
    assert norm.payload is not None
    assert norm.payload["meta"]["candle_close_time"] == "2026-01-01T00:59:59Z"


def test_engineer_latest_does_not_use_future_candle() -> None:
    candles = _candle_rows(50)
    ctx = _engineer_latest(candles, 20, "crypto", None, False)
    assert ctx["signal_reference_price"] == pytest.approx(candles[-1]["close"])


def test_trade_candidate_live_execution_after_close() -> None:
    candles = _candle_rows(100)
    cand = TradeCandidate.from_message(
        {
            "symbol": "ETHUSDT",
            "interval": "1h",
            "asset_class": "crypto",
            "candles": candles,
            "meta": {
                "current_price": candles[-1]["close"],
                "composite_score": 0.8,
                "features": {"current_price": candles[-1]["close"]},
                "candle_close_time": "2026-01-01T00:59:59Z",
            },
        }
    )
    timing = resolve_execution_timing(cand.candles, cand.meta, slippage_bps=5.0)
    from ae_brain.execution.timing import _to_ms

    assert _to_ms(timing.execution_time) >= _to_ms(timing.signal_candle_close_time)
    assert timing.execution_price > timing.signal_reference_price
