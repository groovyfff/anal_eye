from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.continuous_test_publisher import run_continuous_test_publisher


def _config(*, continuous: bool, symbols: list[str]) -> ServiceConfig:
    return ServiceConfig(
        enabled=True,
        symbols=symbols,
        timeframe="1h",
        market="futures",
        wss_base_url="wss://fstream.binance.com/ws",
        reconnect_delay_sec=5,
        min_candles=3,
        bootstrap_limit=10,
        rest_base_url="https://fapi.binance.com",
        publish_mode="throttled",
        throttle_sec=0,
        publish_on_candle_close=False,
        publish_on_every_update=False,
        continuous_test_mode=continuous,
        emit_interval_ms=50,
        emit_round_robin=True,
        emit_require_min_candles=True,
    )


def _seed(buffer: CandleBuffer, symbol: str, n: int) -> None:
    for i in range(n):
        buffer.upsert(
            symbol,
            Candle(timestamp=i, open=1, high=2, low=0.5, close=1.5, volume=1, closed=True),
        )


def test_continuous_mode_disabled_exits_immediately() -> None:
    publisher = AsyncMock()
    stop = asyncio.Event()
    stop.set()
    asyncio.run(
        run_continuous_test_publisher(
            config=_config(continuous=False, symbols=["ETHUSDT"]),
            buffer=CandleBuffer(),
            publisher=publisher,
            stop_event=stop,
        )
    )
    publisher.publish_candidate.assert_not_called()


def test_continuous_mode_round_robin_preserves_symbol() -> None:
    buffer = CandleBuffer()
    _seed(buffer, "ETHUSDT", 5)
    _seed(buffer, "SOLUSDT", 5)

    publisher = AsyncMock()
    stop = asyncio.Event()

    async def _run() -> None:
        task = asyncio.create_task(
            run_continuous_test_publisher(
                config=_config(continuous=True, symbols=["ETHUSDT", "SOLUSDT"]),
                buffer=buffer,
                publisher=publisher,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0.16)
        stop.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert publisher.publish_candidate.await_count >= 2
    symbols = [call.args[0]["symbol"] for call in publisher.publish_candidate.await_args_list]
    assert "ETHUSDT" in symbols
    assert "SOLUSDT" in symbols
    assert "BTCUSDT" not in symbols


def test_continuous_mode_skips_not_ready_symbol() -> None:
    buffer = CandleBuffer()
    _seed(buffer, "ETHUSDT", 5)

    publisher = AsyncMock()

    async def _run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_continuous_test_publisher(
                config=_config(continuous=True, symbols=["SOLUSDT", "ETHUSDT"]),
                buffer=buffer,
                publisher=publisher,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0.12)
        stop.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    published = [call.args[0]["symbol"] for call in publisher.publish_candidate.await_args_list]
    assert published
    assert all(sym == "ETHUSDT" for sym in published)
