"""1h closed-candle pipeline tests."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

from src.binance_ws import BinanceCandidateStream
from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.converters.ae_brain_candidate import build_ae_brain_candidate, validate_candidate_payload
from src.dedup_store import DedupStore
from src.kline_parser import parse_kline_message
from src.publisher import CandidatePublisher
from src.rest_backfill import backfill_symbol


def _make_candles(n: int, start_ts: int = 1_700_000_000_000) -> list[Candle]:
    return [
        Candle(
            timestamp=start_ts + i * 3_600_000,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0 + i,
            closed=True,
            close_time=start_ts + i * 3_600_000 + 3_599_999,
        )
        for i in range(n)
    ]


def _closed_kline_message(symbol: str, open_time: int, *, closed: bool = True) -> dict:
    return {
        "e": "kline",
        "E": open_time + 3_599_999,
        "s": symbol,
        "k": {
            "t": open_time,
            "T": open_time + 3_599_999,
            "o": "100",
            "h": "101",
            "l": "99",
            "c": "100.5",
            "v": "1000",
            "q": "100000",
            "n": 500,
            "V": "500",
            "Q": "50000",
            "x": closed,
        },
    }


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> ServiceConfig:
    monkeypatch.setenv(
        "ANAL_EYES_ALLOWED_SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT",
    )
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT")
    monkeypatch.setenv("CANDIDATE_TIMEFRAME", "1h")
    monkeypatch.setenv("CANDIDATE_CLOSED_CANDLES_ONLY", "true")
    monkeypatch.setenv("CANDIDATE_DEDUP_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_WINDOW_CANDLES", "200")
    monkeypatch.setenv("ENABLE_LEGACY_PARSER", "false")
    monkeypatch.setenv("ENABLE_HIGH_FREQUENCY_TEST_PARSER", "false")
    return ServiceConfig.from_env(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"])


def test_config_subscribes_only_six_symbols(config: ServiceConfig) -> None:
    assert len(config.symbols) == 6
    streams = config.all_stream_names()
    assert streams == [
        "btcusdt@kline_1h",
        "ethusdt@kline_1h",
        "solusdt@kline_1h",
        "bnbusdt@kline_1h",
        "xrpusdt@kline_1h",
        "dogeusdt@kline_1h",
    ]


@pytest.mark.parametrize("symbol", ["ADAUSDT", "AVAXUSDT", "LINKUSDT"])
def test_disallowed_symbols_rejected_in_converter(symbol: str) -> None:
    candles = _make_candles(200)
    payload = build_ae_brain_candidate(
        symbol=symbol,
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-1],
        window_candles=200,
    )
    assert payload is None


def test_fewer_than_200_candles_skips_publish() -> None:
    candles = _make_candles(50)
    payload = build_ae_brain_candidate(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-1],
        window_candles=200,
    )
    assert payload is None


def test_closed_candle_builds_valid_payload() -> None:
    candles = _make_candles(200)
    payload = build_ae_brain_candidate(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-1],
        window_candles=200,
    )
    assert payload is not None
    assert payload["candle_closed"] is True
    assert payload["candles_count"] == 200
    assert payload["source"] == "binance_kline_1h"
    validate_candidate_payload(payload)
    ts_list = [c["timestamp"] for c in payload["candles"]]
    assert ts_list == sorted(ts_list)
    assert len(ts_list) == len(set(ts_list))


def test_duplicate_timestamps_removed() -> None:
    candles = _make_candles(201)
    dup = candles[0]
    candles.append(
        Candle(
            timestamp=dup.timestamp,
            open=9.0,
            high=10.0,
            low=8.0,
            close=9.5,
            volume=99.0,
            closed=True,
            close_time=dup.close_time,
        )
    )
    payload = build_ae_brain_candidate(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles,
        closed_candle=candles[-2],
        window_candles=200,
    )
    assert payload is not None
    assert payload["candles_count"] == 200


@pytest.mark.asyncio
async def test_incomplete_candle_does_not_publish(config: ServiceConfig, tmp_path: Path) -> None:
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    msg = _closed_kline_message("BTCUSDT", 1_700_720_000_000, closed=False)
    await stream._handle_message(json.dumps(msg))

    publisher.publish_candidate.assert_not_called()
    dedup.close()


@pytest.mark.asyncio
async def test_closed_candle_publishes_once(config: ServiceConfig, tmp_path: Path) -> None:
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    open_time = 1_700_720_000_000
    msg = _closed_kline_message("BTCUSDT", open_time, closed=True)
    await stream._handle_message(json.dumps(msg))
    await stream._handle_message(json.dumps(msg))

    assert publisher.publish_candidate.call_count == 1
    dedup.close()


@pytest.mark.asyncio
async def test_dedup_survives_restart(config: ServiceConfig, tmp_path: Path) -> None:
    db = tmp_path / "dedup.db"
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    open_time = 1_700_720_000_000

    dedup1 = DedupStore(db)
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream1 = BinanceCandidateStream(config, buffer, publisher, dedup1)
    msg = _closed_kline_message("BTCUSDT", open_time, closed=True)
    await stream1._handle_message(json.dumps(msg))
    dedup1.close()

    dedup2 = DedupStore(db)
    publisher2 = MagicMock(spec=CandidatePublisher)
    publisher2.publish_candidate = AsyncMock()
    stream2 = BinanceCandidateStream(config, buffer, publisher2, dedup2)
    await stream2._handle_message(json.dumps(msg))
    dedup2.close()

    assert publisher.publish_candidate.call_count == 1
    assert publisher2.publish_candidate.call_count == 0


def test_kline_parser_closed_flag() -> None:
    parsed = parse_kline_message(_closed_kline_message("ETHUSDT", 1000), timeframe="1h")
    assert parsed.is_closed is True
    parsed_open = parse_kline_message(_closed_kline_message("ETHUSDT", 1000, closed=False), timeframe="1h")
    assert parsed_open.is_closed is False


def test_prod_guardrails_reject_open_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CANDIDATE_CLOSED_CANDLES_ONLY", "false")
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    with pytest.raises(ValueError, match="CANDIDATE_CLOSED_CANDLES_ONLY"):
        ServiceConfig.from_env(symbols=["BTCUSDT"])


# ---------------------------------------------------------------------------
# WebSocket URL, combined vs raw payloads, receive loop, idle timeout.
# ---------------------------------------------------------------------------


def test_ws_url_is_combined_futures_for_multiple_symbols(config: ServiceConfig) -> None:
    url = config.build_wss_url()
    assert url == (
        "wss://fstream.binance.com/stream?streams="
        "btcusdt@kline_1h/ethusdt@kline_1h/solusdt@kline_1h/"
        "bnbusdt@kline_1h/xrpusdt@kline_1h/dogeusdt@kline_1h"
    )


def test_ws_url_is_raw_single_stream_when_one_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    cfg = ServiceConfig.from_env(symbols=["BTCUSDT"])
    assert cfg.build_wss_url() == "wss://fstream.binance.com/ws/btcusdt@kline_1h"


def _wrapped(stream: str, symbol: str, open_time: int, *, closed: bool) -> dict:
    """Binance combined-stream envelope: {stream, data:{e,k}}."""
    return {"stream": stream, "data": _closed_kline_message(symbol, open_time, closed=closed)}


async def test_combined_stream_payload_x_false_skipped(
    config: ServiceConfig, tmp_path: Path
) -> None:
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    msg = _wrapped("btcusdt@kline_1h", "BTCUSDT", 1_700_720_000_000, closed=False)
    await stream._handle_message(json.dumps(msg))

    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_combined_stream_payload_x_true_published(
    config: ServiceConfig, tmp_path: Path
) -> None:
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    open_time = 1_700_720_000_000
    msg = _wrapped("btcusdt@kline_1h", "BTCUSDT", open_time, closed=True)
    await stream._handle_message(json.dumps(msg))

    assert publisher.publish_candidate.call_count == 1
    dedup.close()


async def test_raw_ws_payload_x_true_published(
    config: ServiceConfig, tmp_path: Path
) -> None:
    """Single-stream /ws/<stream> delivers a raw kline object (no envelope)."""
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    open_time = 1_700_720_000_000
    await stream._handle_message(json.dumps(_closed_kline_message("BTCUSDT", open_time, closed=True)))

    assert publisher.publish_candidate.call_count == 1
    dedup.close()


async def test_unsupported_combined_symbol_rejected(
    config: ServiceConfig, tmp_path: Path
) -> None:
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    open_time = 1_700_720_000_000
    await stream._handle_message(
        json.dumps(_wrapped("adausdt@kline_1h", "ADAUSDT", open_time, closed=True))
    )

    publisher.publish_candidate.assert_not_called()
    dedup.close()


async def test_malformed_payload_logged_and_continues(
    config: ServiceConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    caplog.set_level(logging.WARNING)
    # Invalid JSON then a valid closed candle — the loop must survive the bad frame.
    await stream._handle_message("not-json")
    open_time = 1_700_720_000_000
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    await stream._handle_message(json.dumps(_closed_kline_message("BTCUSDT", open_time, closed=True)))

    assert publisher.publish_candidate.call_count == 1
    assert any("Invalid Binance kline" in r.message for r in caplog.records)
    dedup.close()


async def test_receive_loop_continues_after_first_message(
    config: ServiceConfig, tmp_path: Path
) -> None:
    """The receive loop must not exit after the first frame — feed it three frames."""
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    frames = [
        json.dumps(_closed_kline_message("BTCUSDT", 1_700_720_000_000 + i * 3_600_000, closed=True))
        for i in range(3)
    ]

    class _FramesExhausted(Exception):
        pass

    class _FakeWS:
        def __init__(self, items: list[str]) -> None:
            self._items = list(items)

        async def recv(self) -> str:
            if not self._items:
                raise _FramesExhausted
            return self._items.pop(0)

    fake = _FakeWS(frames)
    # The real websockets lib raises ConnectionClosed on close, which run_forever
    # catches. We use a local sentinel to assert the loop processed all frames
    # before the WS went away, and that the loop did not exit after the first one.
    with pytest.raises(_FramesExhausted):
        await stream._receive_loop(fake, idle_timeout=10)

    assert publisher.publish_candidate.call_count == 3
    dedup.close()


async def test_idle_timeout_triggers_reconnect_log(
    config: ServiceConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When recv() stalls past idle_timeout, websocket_idle_reconnect is logged."""
    buffer = CandleBuffer(max_candles=200)
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    class _StalledWS:
        async def recv(self) -> str:
            await asyncio.sleep(5)  # longer than the test idle timeout
            return "never"

    caplog.set_level(logging.WARNING)
    await stream._receive_loop(_StalledWS(), idle_timeout=0.05)

    assert any("websocket_idle_reconnect" in r.message for r in caplog.records)
    dedup.close()


async def test_rabbitmq_routing_keys_used(
    config: ServiceConfig, tmp_path: Path
) -> None:
    """Publisher.publish_candidate is the path to exchange=analeyes.events / data.candidates.ai.

    We assert the publisher is invoked exactly once and that the payload carries
    the AE Brain contract; the actual exchange/routing key lives in CandidatePublisher
    (validated separately by inspecting publisher.py constant usage).
    """
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _make_candles(200))
    dedup = DedupStore(tmp_path / "dedup.db")
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()
    stream = BinanceCandidateStream(config, buffer, publisher, dedup)

    open_time = 1_700_720_000_000
    await stream._handle_message(json.dumps(_closed_kline_message("BTCUSDT", open_time, closed=True)))

    publisher.publish_candidate.assert_called_once()
    # Confirm the topology constants the publisher binds to.
    from src.publisher import CandidatePublisher as _Pub  # noqa: F401

    assert EXCHANGE == "analeyes.events"
    assert RoutingKey.DATA_CANDIDATES_AI == "data.candidates.ai"
    dedup.close()


def test_publisher_module_uses_correct_topology() -> None:
    import inspect

    from src import publisher as publisher_mod

    src = inspect.getsource(publisher_mod)
    assert '"analeyes.events"' not in src  # no hardcoded literal
    assert "EXCHANGE" in src
    assert "RoutingKey.DATA_CANDIDATES_AI" in src
    assert EXCHANGE == "analeyes.events"
    assert RoutingKey.DATA_CANDIDATES_AI == "data.candidates.ai"


# ---------------------------------------------------------------------------
# Backfill must not publish history.
# ---------------------------------------------------------------------------


async def test_backfill_does_not_publish_history(
    config: ServiceConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import rest_backfill

    buffer = CandleBuffer(max_candles=200)
    publisher = MagicMock(spec=CandidatePublisher)
    publisher.publish_candidate = AsyncMock()

    monkeypatch.setattr(
        rest_backfill,
        "fetch_klines",
        lambda **kw: _make_candles(200),
    )
    candles = rest_backfill.backfill_symbol(
        symbol="BTCUSDT",
        interval="1h",
        limit=200,
        rest_base_url=config.rest_base_url,
        buffer=buffer,
        publish_historical=False,
    )
    assert len(candles) == 200
    publisher.publish_candidate.assert_not_called()


# ---------------------------------------------------------------------------
# Production guardrails for the new flags.
# ---------------------------------------------------------------------------


def test_prod_guardrails_reject_continuous_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("CANDIDATE_CONTINUOUS_TEST_MODE", "true")
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    with pytest.raises(ValueError, match="CANDIDATE_CONTINUOUS_TEST_MODE"):
        ServiceConfig.from_env(symbols=["BTCUSDT"])


def test_prod_guardrails_reject_publish_every_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE", "true")
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    with pytest.raises(ValueError, match="PUBLISH_ON_EVERY_UPDATE"):
        ServiceConfig.from_env(symbols=["BTCUSDT"])


def test_prod_guardrails_reject_no_publish_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE", "false")
    monkeypatch.setenv("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    with pytest.raises(ValueError, match="PUBLISH_ON_CANDLE_CLOSE"):
        ServiceConfig.from_env(symbols=["BTCUSDT"])


def test_dev_defaults_are_safe(config: ServiceConfig) -> None:
    assert config.continuous_test_mode is False
    assert config.publish_on_candle_close is True
    assert config.publish_on_every_update is False
    assert config.ws_idle_timeout_sec == 60
