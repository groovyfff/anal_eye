"""Tests for the REST closed-candle fallback poller."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

from src.binance_ws import BinanceCandidateStream
from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.dedup_store import DedupStore
from src.rest_closed_candle_poller import RestClosedCandlePoller, latest_closed_candle

_HISTORICAL = 200
_HOUR_MS = 3_600_000


def _gen_candles(n_closed: int, *, last_close_ts: int, with_open: bool = True) -> list[Candle]:
    """Build ``n_closed`` closed candles ending at ``last_close_ts`` + optional open candle."""
    last_open = last_close_ts - (_HOUR_MS - 1)  # close_time of the last closed candle
    start_open = last_open - (n_closed - 1) * _HOUR_MS
    out: list[Candle] = []
    for i in range(n_closed):
        ot = start_open + i * _HOUR_MS
        out.append(
            Candle(
                timestamp=ot,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
                closed=True,
                close_time=ot + _HOUR_MS - 1,
            )
        )
    if with_open:
        ot = last_open + _HOUR_MS
        out.append(
            Candle(
                timestamp=ot,
                open=200.0,
                high=201.0,
                low=199.0,
                close=200.5,
                volume=500.0,
                closed=False,
                close_time=ot + _HOUR_MS - 1,
            )
        )
    return out


def _single_symbol_config(monkeypatch: pytest.MonkeyPatch, symbol: str = "BTCUSDT") -> ServiceConfig:
    """Single-symbol config keeps poller tests deterministic (one fetch -> one publish)."""
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", symbol)
    monkeypatch.setenv("SYMBOLS", symbol)
    monkeypatch.setenv("CANDIDATE_TIMEFRAME", "1h")
    monkeypatch.setenv("CANDIDATE_CLOSED_CANDLES_ONLY", "true")
    monkeypatch.setenv("CANDIDATE_DEDUP_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_WINDOW_CANDLES", "200")
    return ServiceConfig.from_env(symbols=[symbol])


def _make_stream(
    config: ServiceConfig, buffer: CandleBuffer, dedup: DedupStore
) -> tuple[BinanceCandidateStream, MagicMock]:
    publisher = MagicMock()
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)
    return stream, publisher


# ---------------------------------------------------------------------------
# latest_closed_candle helper
# ---------------------------------------------------------------------------


def test_latest_closed_candle_excludes_open_candle() -> None:
    now_ms = 1_700_800_000_000
    candles = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=True)
    latest = latest_closed_candle(candles, now_ms=now_ms)
    assert latest is not None
    assert latest.closed is True
    assert latest.close_time < now_ms


def test_latest_closed_candle_returns_none_when_all_open() -> None:
    now_ms = 1_700_800_000_000
    open_only = Candle(
        timestamp=now_ms,
        open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0,
        closed=False, close_time=now_ms + _HOUR_MS - 1,
    )
    assert latest_closed_candle([open_only], now_ms=now_ms) is None


# ---------------------------------------------------------------------------
# Publish behavior
# ---------------------------------------------------------------------------


async def test_rest_fallback_does_not_publish_open_candle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000

    open_only = Candle(
        timestamp=now_ms, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0,
        closed=False, close_time=now_ms + _HOUR_MS - 1,
    )
    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=lambda **_: [open_only],
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
    )
    await poller._poll_once()

    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_rest_fallback_publishes_one_new_closed_candle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000
    candles = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=lambda **_: list(candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,
    )
    await poller._poll_once()

    publisher.publish_candidate.assert_called_once()
    dedup.close()


async def test_rest_fallback_dedups_duplicate_closed_candle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000
    candles = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=lambda **_: list(candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,
    )
    await poller._poll_once()
    await poller._poll_once()  # duplicate

    assert publisher.publish_candidate.call_count == 1
    dedup.close()


async def test_rest_fallback_and_ws_share_dedup_no_double_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WS delivers a closed candle first; REST polling the same candle must skip."""
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000

    history = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=False)
    buffer.load_bootstrap("BTCUSDT", history)

    ws_candle = history[-1]
    status_ws = await stream.publish_closed_candle_for("BTCUSDT", ws_candle)
    assert status_ws == "published"

    # REST returns the SAME closed candles (so the latest-closed timestamp
    # matches what WS already published) plus the currently-open one.
    rest_candles = list(history) + [
        Candle(
            timestamp=ws_candle.timestamp + _HOUR_MS,
            open=200.0, high=201.0, low=199.0, close=200.5, volume=500.0,
            closed=False, close_time=ws_candle.timestamp + 2 * _HOUR_MS - 1,
        )
    ]
    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: True,  # WS healthy — but always_on default True
        fetcher=lambda **_: list(rest_candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,  # WS candle close_time is strictly in the past
    )
    await poller._poll_once()

    # Total publishes across both paths must be exactly 1 (shared dedup).
    assert publisher.publish_candidate.call_count == 1
    dedup.close()


async def test_rest_fallback_unsupported_symbol_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ADAUSDT is not in the production universe; converter returns None -> no publish.

    Keep the real 6-symbol universe (so ADAUSDT is genuinely unsupported) while
    telling the poller to fetch ADAUSDT.
    """
    monkeypatch.setenv(
        "ANAL_EYES_ALLOWED_SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT",
    )
    monkeypatch.setenv("SYMBOLS", "ADAUSDT")  # poller will fetch this
    monkeypatch.setenv("CANDIDATE_TIMEFRAME", "1h")
    monkeypatch.setenv("CANDIDATE_CLOSED_CANDLES_ONLY", "true")
    monkeypatch.setenv("CANDIDATE_DEDUP_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_WINDOW_CANDLES", "200")
    # resolve_production_symbols intersects SYMBOLS with the universe -> [] would
    # be empty; to force the poller to actually iterate ADAUSDT, build the config
    # with the production universe then swap symbols via dataclasses.replace.
    base = ServiceConfig.from_env(symbols=["BTCUSDT"])  # valid base
    config = dataclasses.replace(base, symbols=["ADAUSDT"])

    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000
    candles = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=lambda **_: list(candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,
    )
    await poller._poll_once()

    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_rest_fallback_under_200_candles_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000
    # Only 50 closed candles — converter requires 200.
    candles = _gen_candles(50, last_close_ts=now_ms, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=lambda **_: list(candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,
    )
    await poller._poll_once()

    publisher.publish_candidate.assert_not_called()
    dedup.close()


def test_rest_fallback_uses_correct_rabbitmq_routing() -> None:
    # The poller funnels through stream.publish_closed_candle_for -> publisher,
    # which uses EXCHANGE + RoutingKey.DATA_CANDIDATES_AI.
    assert EXCHANGE == "analeyes.events"
    assert RoutingKey.DATA_CANDIDATES_AI == "data.candidates.ai"


async def test_rest_fallback_skips_polling_when_ws_healthy_and_not_always_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = dataclasses.replace(_single_symbol_config(monkeypatch), rest_fallback_always_on=False)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)

    fetched = {"count": 0}

    def _fetch(**_):
        fetched["count"] += 1
        return _gen_candles(_HISTORICAL, last_close_ts=1_700_800_000_000, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: True,  # WS healthy
        fetcher=_fetch,
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
    )
    await poller._poll_once()

    assert fetched["count"] == 0, "must not poll REST when WS healthy and always_on=False"
    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_rest_fallback_error_does_not_crash_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)

    def _boom(**_):
        raise RuntimeError("network down")

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,
        fetcher=_boom,
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
    )
    # A single poll iteration must not raise; run_forever swallows errors.
    await poller._poll_once()
    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_rest_fallback_publishes_when_ws_dead_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: WS never received (unhealthy) -> REST must publish the closed candle."""
    config = _single_symbol_config(monkeypatch)
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    stream, publisher = _make_stream(config, buffer, dedup)
    now_ms = 1_700_800_000_000
    candles = _gen_candles(_HISTORICAL, last_close_ts=now_ms, with_open=True)

    poller = RestClosedCandlePoller(
        config, buffer, stream,
        ws_health_getter=lambda: False,  # WS dead
        fetcher=lambda **_: list(candles),
        sleep_fn=lambda *_a, **_k: asyncio.sleep(0),
        now_fn=lambda: now_ms + 1,
    )
    await poller._poll_once()

    assert publisher.publish_candidate.call_count == 1
    payload = publisher.publish_candidate.call_args.args[0]
    assert payload["symbol"] == "BTCUSDT"
    assert payload["candle_closed"] is True
    assert payload["candles_count"] == _HISTORICAL
    dedup.close()
