from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from shared.symbol_universe import default_allowed_symbols, is_symbol_allowed
from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.converters.ae_brain_candidate import build_ae_brain_candidate, ms_to_iso_utc
from src.dedup_store import DedupStore
from src.kline_parser import parse_kline_message
from src.publisher import CandidatePublisher
from src.rest_backfill import fetch_optional_market_fields, gap_recover_symbol

logger = logging.getLogger(__name__)

GapRecoverFn = Callable[[], Awaitable[None]]


class BinanceCandidateStream:
    """USD-M Futures 1h kline stream — publishes one candidate per closed candle."""

    def __init__(
        self,
        config: ServiceConfig,
        buffer: CandleBuffer,
        publisher: CandidatePublisher,
        dedup: DedupStore,
        *,
        optional_features: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._buffer = buffer
        self._publisher = publisher
        self._dedup = dedup
        self._optional_features = optional_features or {}
        self._connected_once = False
        # Monotonic timestamp of the last received WS frame, for health signaling
        # to the REST fallback poller. None until the first frame arrives.
        self._last_message_monotonic: float | None = None

    def last_received_monotonic(self) -> float | None:
        """Last monotonic time a WS frame was received (None if never)."""
        return self._last_message_monotonic

    def ws_is_healthy(self, *, now: float | None = None) -> bool:
        """True when a frame has been received within the idle timeout window."""
        if self._last_message_monotonic is None:
            return False
        current = now if now is not None else time.monotonic()
        return (current - self._last_message_monotonic) <= self._config.ws_idle_timeout_sec

    async def run_forever(self) -> None:
        delay = self._config.reconnect_delay_sec
        idle_timeout = self._config.ws_idle_timeout_sec
        while True:
            if self._connected_once:
                await self._recover_gaps()
            wss_url = self._config.build_wss_url()
            stream_label = ",".join(self._config.all_stream_names())
            logger.info(
                "websocket_url url=%s streams=%s timeframe=%s source=binance_futures_api",
                wss_url,
                stream_label,
                self._config.timeframe,
            )
            try:
                async with websockets.connect(
                    wss_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=2**20,
                ) as ws:
                    event = "websocket_reconnected" if self._connected_once else "websocket_connected"
                    logger.info(
                        "%s streams=%s timeframe=%s source=binance_futures_api",
                        event,
                        stream_label,
                        self._config.timeframe,
                    )
                    self._connected_once = True
                    await self._receive_loop(ws, idle_timeout)
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                reason = f"code={exc.code} reason={exc.reason or 'connection closed'}"
                logger.warning(
                    "websocket_error reason=%s reconnect_in=%ss", reason, delay
                )
            except asyncio.TimeoutError:
                # Application-level idle: TCP/PONG alive but no kline frame seen.
                logger.warning(
                    "websocket_error reason=idle_timeout reconnect_in=%ss", delay
                )
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__
                logger.warning(
                    "websocket_error reason=%s reconnect_in=%ss", reason, delay
                )
            await asyncio.sleep(delay)

    async def _receive_loop(self, ws: Any, idle_timeout: int) -> None:
        """Await messages continuously; reconnect when no frame arrives within idle_timeout.

        Uses ``wait_for(ws.recv(), timeout)`` instead of ``async for`` so a
        silently-connected-but-data-starved stream (server answers PONGs but
        stops emitting kline frames) is detected and recovered instead of
        hanging forever. Must run until shutdown or disconnect.
        """
        while True:
            try:
                raw_message = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "websocket_idle_reconnect reason=no_message_within_%ss streams=%s",
                    idle_timeout,
                    ",".join(self._config.all_stream_names()),
                )
                # Break out of the loop so run_forever reconnects (with gap recovery).
                return
            logger.debug(
                "websocket_message_received bytes=%s",
                len(raw_message) if raw_message is not None else 0,
            )
            self._last_message_monotonic = time.monotonic()
            await self._handle_message(raw_message)

    async def _recover_gaps(self) -> None:
        loop = asyncio.get_running_loop()
        for symbol in self._config.symbols:
            last_close = self._buffer.last_close_time(symbol)
            await loop.run_in_executor(
                None,
                lambda sym=symbol, lc=last_close: gap_recover_symbol(
                    symbol=sym,
                    interval=self._config.timeframe,
                    rest_base_url=self._config.rest_base_url,
                    buffer=self._buffer,
                    last_close_time=lc,
                    window_candles=self._config.window_candles,
                ),
            )

    async def _handle_message(self, raw_message: str | bytes) -> None:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            message: dict[str, Any] = json.loads(raw_message)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Invalid Binance kline reason=invalid json: %s", exc)
            return

        default_stream = self._config.all_stream_names()[0] if len(self._config.symbols) == 1 else None
        try:
            parsed = parse_kline_message(
                message,
                timeframe=self._config.timeframe,
                default_stream=default_stream,
            )
        except ValueError as exc:
            logger.warning("Invalid Binance kline reason=%s", exc)
            return

        logger.info(
            "kline_update_received symbol=%s open_time=%s closed=%s stream=%s",
            parsed.symbol,
            parsed.open_time,
            parsed.is_closed,
            parsed.raw_stream,
        )

        if not is_symbol_allowed(parsed.symbol, default_allowed_symbols()):
            logger.info(
                "candidate_rejected_symbol symbol=%s allowed=%s",
                parsed.symbol,
                ",".join(sorted(default_allowed_symbols())),
            )
            return

        candle = Candle(
            timestamp=parsed.open_time,
            open=parsed.open,
            high=parsed.high,
            low=parsed.low,
            close=parsed.close,
            volume=parsed.volume,
            closed=parsed.is_closed,
            close_time=parsed.close_time,
            quote_volume=parsed.quote_volume,
            trades_count=parsed.trades_count,
            taker_buy_base_volume=parsed.taker_buy_base_volume,
            taker_buy_quote_volume=parsed.taker_buy_quote_volume,
        )
        self._buffer.upsert(parsed.symbol, candle)

        if self._config.closed_candles_only and not parsed.is_closed:
            logger.info(
                "candle_not_closed_skipped symbol=%s timeframe=%s candle_open_time=%s",
                parsed.symbol,
                self._config.timeframe,
                ms_to_iso_utc(parsed.open_time),
            )
            return

        if not parsed.is_closed:
            return

        logger.info(
            "candle_closed_received symbol=%s timeframe=%s candle_open_time=%s",
            parsed.symbol,
            self._config.timeframe,
            ms_to_iso_utc(parsed.open_time),
        )
        await self.publish_closed_candle_for(parsed.symbol, candle)

    async def publish_closed_candle_for(self, symbol: str, closed_candle: Candle) -> str:
        """Shared publish path used by both WebSocket and REST fallback.

        Enforces the SQLite dedup guard (key = symbol:timeframe:candle_open_time),
        builds the AE Brain candidate via the converter, publishes exactly once,
        and marks the dedup store. Returns ``"published"`` when published,
        ``"dedup_skipped"`` when the dedup guard short-circuited, ``"not_ready"``
        when the converter returned None (e.g. < 200 candles), or ``"failed"``
        when the publisher raised. Idempotent and safe to call concurrently with
        the WebSocket path — the dedup store guarantees no double-publish.
        """
        if self._config.dedup_enabled and self._dedup.was_published(
            symbol, self._config.timeframe, closed_candle.timestamp
        ):
            logger.info(
                "candidate_dedup_skipped symbol=%s timeframe=%s candle_open_time=%s candle_close_time=%s",
                symbol,
                self._config.timeframe,
                ms_to_iso_utc(closed_candle.timestamp),
                ms_to_iso_utc(closed_candle.close_time or closed_candle.timestamp),
            )
            return "dedup_skipped"
        if self._config.dedup_enabled and self._dedup.was_published(
            symbol, self._config.timeframe, closed_candle.timestamp
        ):
            logger.info(
                "candidate_dedup_skipped symbol=%s timeframe=%s candle_open_time=%s candle_close_time=%s",
                symbol,
                self._config.timeframe,
                ms_to_iso_utc(closed_candle.timestamp),
                ms_to_iso_utc(closed_candle.close_time or closed_candle.timestamp),
            )
            return

        optional = self._optional_features.get(symbol.upper(), {})
        payload = build_ae_brain_candidate(
            symbol=symbol,
            timeframe=self._config.timeframe,
            candles=self._buffer.candles(symbol),
            closed_candle=closed_candle,
            market=self._config.market,
            optional_features=optional,
            window_candles=self._config.window_candles,
        )
        if payload is None:
            return "not_ready"

        try:
            await self._publisher.publish_candidate(payload)
        except ValueError as exc:
            logger.warning(
                "candidate_publish_failed symbol=%s reason=%s exchange=%s routing_key=%s",
                symbol,
                exc,
                EXCHANGE,
                RoutingKey.DATA_CANDIDATES_AI,
            )
            return "failed"
        except Exception as exc:
            logger.error(
                "candidate_publish_failed symbol=%s err=%s exchange=%s routing_key=%s",
                symbol,
                exc,
                EXCHANGE,
                RoutingKey.DATA_CANDIDATES_AI,
            )
            return "failed"

        if self._config.dedup_enabled:
            self._dedup.mark_published(
                symbol,
                self._config.timeframe,
                closed_candle.timestamp,
                closed_candle.close_time or closed_candle.timestamp,
            )

        logger.info(
            "candidate_publish_allowed symbol=%s timeframe=%s candle_open_time=%s candle_close_time=%s "
            "candles_count=%s exchange=%s routing_key=%s",
            symbol,
            self._config.timeframe,
            payload["candle_open_time"],
            payload["candle_close_time"],
            payload["candles_count"],
            EXCHANGE,
            RoutingKey.DATA_CANDIDATES_AI,
        )
        return "published"
