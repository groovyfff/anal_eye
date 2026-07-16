"""Binance candidate publisher symbol filter tests."""

from __future__ import annotations

import pytest

from shared.symbol_universe import is_symbol_allowed, default_allowed_symbols

from src.converters.ae_brain_candidate import build_ae_brain_candidate
from src.candle_buffer import Candle
from src.publisher import CandidatePublisher


@pytest.fixture
def publisher() -> CandidatePublisher:
    return CandidatePublisher("amqp://guest:guest@localhost:5672/", user="guest", vhost="/")


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


@pytest.mark.asyncio
async def test_adausdt_rejected_before_publish(publisher: CandidatePublisher) -> None:
    with pytest.raises(ValueError, match="symbol_not_allowed:ADAUSDT"):
        await publisher.publish_candidate({"symbol": "ADAUSDT", "candle_closed": True, "candles": []})


def test_btcusdt_allowed_in_universe() -> None:
    assert is_symbol_allowed("BTCUSDT", default_allowed_symbols())


@pytest.mark.asyncio
async def test_btcusdt_valid_payload_passes_gate(publisher: CandidatePublisher) -> None:
    candles = _candles(200)
    payload = build_ae_brain_candidate(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-1],
        window_candles=200,
    )
    assert payload is not None
    published: list[dict] = []

    async def _fake_publish(exchange: str, routing_key: str, body: str) -> None:
        import json

        published.append(json.loads(body))

    publisher._client.publish_async = _fake_publish  # type: ignore[method-assign]
    await publisher.publish_candidate(payload)
    assert published[0]["symbol"] == "BTCUSDT"
