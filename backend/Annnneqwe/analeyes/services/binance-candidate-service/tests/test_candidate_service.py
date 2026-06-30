from __future__ import annotations

import math

import pytest

from src.candle_buffer import Candle, CandleBuffer
from src.candidate_builder import assert_no_forbidden_fields, build_candidate_payload
from src.indicators import compute_features
from src.kline_parser import parse_kline_message


def _make_candles(n: int, start: float = 60000.0) -> list[Candle]:
    candles: list[Candle] = []
    price = start
    for i in range(n):
        price += math.sin(i / 7.0) * 50.0
        candles.append(
            Candle(
                timestamp=1_700_000_000_000 + i * 3_600_000,
                open=price - 10,
                high=price + 20,
                low=price - 20,
                close=price,
                volume=100.0 + i,
                closed=True,
            )
        )
    return candles


def test_parse_kline_message() -> None:
    message = {
        "e": "kline",
        "E": 1000,
        "s": "BTCUSDT",
        "k": {
            "t": 900,
            "T": 1000,
            "o": "1",
            "h": "2",
            "l": "0.5",
            "c": "1.5",
            "v": "10",
            "x": False,
        },
    }
    parsed = parse_kline_message(message, timeframe="1h", default_stream="btcusdt@kline_1h")
    assert parsed.symbol == "BTCUSDT"
    assert parsed.close == 1.5
    assert parsed.is_closed is False


def test_candle_buffer_upsert_and_trim() -> None:
    buffer = CandleBuffer(max_candles=3)
    for i in range(5):
        buffer.upsert(
            "BTCUSDT",
            Candle(timestamp=i, open=1, high=2, low=0.5, close=1.5, volume=1, closed=True),
        )
    assert buffer.count("BTCUSDT") == 3
    assert buffer.candles("BTCUSDT")[0].timestamp == 2


def test_features_with_100_candles() -> None:
    features = compute_features(_make_candles(100))
    assert features["current_price"] is not None
    assert features["rsi"] is not None
    assert features["macd"] is not None
    assert features["ema_short"] is not None
    assert features["ema_200"] is not None


def test_candidate_schema_and_no_forbidden_fields() -> None:
    candles = _make_candles(100)
    payload = build_candidate_payload(
        symbol="BTCUSDT",
        market="futures",
        timeframe="1h",
        event_time=1_700_000_000_000,
        candles=candles,
    )
    assert payload["symbol"] == "BTCUSDT"
    assert payload["asset_class"] == "crypto"
    assert payload["current_price"] > 0
    assert 0.0 <= payload["composite_score"] <= 1.0
    assert len(payload["candles"]) == 100
    assert "features" in payload
    assert_no_forbidden_fields(payload)
    for key in ("decision", "signal_type", "side", "entry_price", "tp_price", "sl_price"):
        assert key not in payload


@pytest.mark.parametrize("symbol", ["ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_candidate_builder_preserves_dynamic_symbol(symbol: str) -> None:
    payload = build_candidate_payload(
        symbol=symbol,
        market="futures",
        timeframe="1h",
        event_time=1_700_000_000_000,
        candles=_make_candles(100, start=3000.0),
    )
    assert payload["symbol"] == symbol


def test_candidate_builder_rejects_missing_symbol() -> None:
    with pytest.raises(ValueError, match="missing_symbol"):
        build_candidate_payload(
            symbol="",
            market="futures",
            timeframe="1h",
            event_time=1,
            candles=_make_candles(10),
        )
