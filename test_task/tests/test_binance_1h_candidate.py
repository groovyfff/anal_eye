"""AE Brain accepts Binance 1h candidate payloads from the new converter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing binance converter from backend tree.
_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend" / "Annnneqwe" / "analeyes"
_BINANCE_SRC = _BACKEND / "services" / "binance-candidate-service"
_SHARED_SRC = _BACKEND / "shared" / "src"
for p in (str(_BINANCE_SRC), str(_SHARED_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ae_brain.messaging.candidate_normalizer import normalize_candidate
from src.candle_buffer import Candle
from src.converters.ae_brain_candidate import build_ae_brain_candidate


def _candles(n: int) -> list[Candle]:
    start = 1_700_000_000_000
    return [
        Candle(
            timestamp=start + i * 3_600_000,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
            closed=True,
            close_time=start + i * 3_600_000 + 3_599_999,
        )
        for i in range(n)
    ]


def test_binance_1h_payload_normalizes_in_ae_brain() -> None:
    candles = _candles(200)
    payload = build_ae_brain_candidate(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-1],
        window_candles=200,
    )
    assert payload is not None
    result = normalize_candidate(payload, min_composite_score=0.0)
    assert result.skip_reason is None
    assert result.payload is not None
    assert result.payload["symbol"] == "BTCUSDT"
    assert result.payload["interval"] == "1h"
    assert len(result.payload["candles"]) == 200


@pytest.mark.parametrize("symbol", ["ADAUSDT", "AVAXUSDT", "LINKUSDT"])
def test_unsupported_symbol_converter_returns_none(symbol: str) -> None:
    candles = _candles(200)
    assert (
        build_ae_brain_candidate(
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            closed_candle=candles[-1],
            window_candles=200,
        )
        is None
    )
