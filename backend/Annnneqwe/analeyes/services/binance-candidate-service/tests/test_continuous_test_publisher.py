from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.continuous_test_publisher import run_continuous_test_publisher


def _config(*, hf_enabled: bool, symbols: list[str], window: int = 3) -> ServiceConfig:
    return ServiceConfig(
        enabled=True,
        symbols=symbols,
        timeframe="1h",
        market="usdm_futures",
        wss_base_url="wss://fstream.binance.com/ws",
        reconnect_delay_sec=5,
        window_candles=window,
        rest_base_url="https://fapi.binance.com",
        closed_candles_only=True,
        dedup_enabled=False,
        min_interval_sec=3600,
        enable_legacy_parser=False,
        enable_high_frequency_test_parser=hf_enabled,
        continuous_test_mode=False,
        publish_on_candle_close=True,
        publish_on_every_update=False,
        backfill_publish_historical=False,
        dedup_db_path="/tmp/test_dedup.db",
        app_env="dev",
        max_candles=window,
    )


def _seed(buffer: CandleBuffer, symbol: str, n: int) -> None:
    for i in range(n):
        buffer.upsert(
            symbol,
            Candle(
                timestamp=1_700_000_000_000 + i * 3_600_000,
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                volume=1,
                closed=True,
                close_time=1_700_000_000_000 + i * 3_600_000 + 3_599_999,
            ),
        )


def test_high_frequency_parser_disabled_exits_immediately() -> None:
    publisher = AsyncMock()
    stop = asyncio.Event()
    stop.set()
    asyncio.run(
        run_continuous_test_publisher(
            config=_config(hf_enabled=False, symbols=["ETHUSDT"]),
            buffer=CandleBuffer(),
            publisher=publisher,
            stop_event=stop,
        )
    )
    publisher.publish_candidate.assert_not_called()


def test_high_frequency_parser_round_robin_preserves_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CANDIDATE_EMIT_INTERVAL_MS", "10")
    buffer = CandleBuffer()
    _seed(buffer, "ETHUSDT", 5)
    _seed(buffer, "SOLUSDT", 5)

    publisher = AsyncMock()
    stop = asyncio.Event()

    async def _run() -> None:
        task = asyncio.create_task(
            run_continuous_test_publisher(
                config=_config(hf_enabled=True, symbols=["ETHUSDT", "SOLUSDT"], window=3),
                buffer=buffer,
                publisher=publisher,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0.35)
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


def test_high_frequency_parser_skips_not_ready_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CANDIDATE_EMIT_INTERVAL_MS", "10")
    buffer = CandleBuffer()
    _seed(buffer, "ETHUSDT", 5)

    publisher = AsyncMock()

    async def _run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_continuous_test_publisher(
                config=_config(hf_enabled=True, symbols=["SOLUSDT", "ETHUSDT"], window=3),
                buffer=buffer,
                publisher=publisher,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0.25)
        stop.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    published = [call.args[0]["symbol"] for call in publisher.publish_candidate.await_args_list]
    assert published
    assert all(sym == "ETHUSDT" for sym in published)
