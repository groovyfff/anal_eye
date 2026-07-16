"""REST closed-candle fallback poller for the candidate pipeline.

When the Binance USD-M Futures WebSocket stops delivering kline frames (e.g. a
silently-dropped combined-stream subscription that still answers PONGs), this
poller keeps the pipeline alive by fetching closed 1h candles from the public
REST endpoint ``/fapi/v1/klines``.

Design contract
---------------
* Shares the **same** ``CandleBuffer``, ``DedupStore``, ``CandidatePublisher``
  and converter as :class:`src.binance_ws.BinanceCandidateStream`.
* Publishes only the **latest fully-closed** candle per poll — never the
  currently-forming candle.
* At most one candidate per ``symbol:timeframe:candle_open_time`` because the
  SQLite dedup store is the single source of truth shared with the WS path.
* Public Binance market data — no API key/secret required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.converters.ae_brain_candidate import ms_to_iso_utc
from src.rest_backfill import fetch_klines

logger = logging.getLogger(__name__)

# 1h candle length in milliseconds.
_HOUR_MS = 3_600_000

WsHealthGetter = Callable[[], bool]


def latest_closed_candle(candles: list[Candle], *, now_ms: int) -> Candle | None:
    """Return the most recent candle whose close_time is strictly in the past.

    A Binance 1h candle with open_time ``t`` closes at ``t + 3_599_999`` ms and
    is "fully closed" once wall-clock time passes ``t + 3_600_000`` ms. The
    currently-forming candle (close_time >= now) is excluded.
    """
    closed = [c for c in candles if c.close_time and c.close_time < now_ms]
    if not closed:
        return None
    return max(closed, key=lambda c: c.timestamp)


class RestClosedCandlePoller:
    """Periodically polls REST klines and publishes new closed candles.

    The poller is safe to run concurrently with the WebSocket stream: both
    paths funnel through ``stream.publish_closed_candle_for`` which is guarded
    by the shared SQLite dedup store, so a candle delivered by both WS and REST
    is published exactly once.
    """

    def __init__(
        self,
        config: ServiceConfig,
        buffer: CandleBuffer,
        stream: Any,
        *,
        ws_health_getter: WsHealthGetter,
        fetcher: Callable[..., list[Candle]] | None = None,
        sleep_fn: Callable[[float], Any] | None = None,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        self._config = config
        self._buffer = buffer
        self._stream = stream
        self._ws_health_getter = ws_health_getter
        # Injectable for tests; defaults to the real REST fetcher + asyncio.sleep.
        self._fetcher = fetcher or fetch_klines
        self._sleep = sleep_fn or asyncio.sleep
        self._now_fn = now_fn or (lambda: int(time.time() * 1000))
        self._ws_unhealthy_emitted = False

    async def run_forever(self) -> None:
        poll_sec = self._config.rest_fallback_poll_sec
        logger.info(
            "rest_fallback_started poll_sec=%s symbols=%s always_on=%s ws_idle_timeout_sec=%s "
            "source=binance_futures_rest",
            poll_sec,
            len(self._config.symbols),
            self._config.rest_fallback_always_on,
            self._config.ws_idle_timeout_sec,
        )
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - poller must survive errors
                logger.warning(
                    "rest_fallback_error reason=%s err=%s",
                    exc.__class__.__name__,
                    exc,
                )
            await self._sleep(poll_sec)

    def _emit_ws_health(self, unhealthy: bool) -> None:
        if unhealthy and not self._ws_unhealthy_emitted:
            logger.warning("ws_marked_unhealthy reason=no_ws_frames_within_idle_timeout")
            self._ws_unhealthy_emitted = True
        elif not unhealthy and self._ws_unhealthy_emitted:
            logger.info("ws_marked_healthy reason=ws_frames_received_again")
            self._ws_unhealthy_emitted = False

    async def _poll_once(self) -> None:
        loop = asyncio.get_running_loop()
        now_ms = self._now_fn()
        ws_healthy = self._ws_health_getter()
        ws_unhealthy = not ws_healthy
        self._emit_ws_health(ws_unhealthy)

        # When WS is healthy and always_on is off, skip active polling; the WS
        # path owns delivery and we avoid unnecessary REST traffic. We still log
        # a heartbeat so operators can see the fallback is armed.
        if ws_healthy and not self._config.rest_fallback_always_on:
            logger.info(
                "rest_fallback_poll skipped=true reason=ws_healthy_and_not_always_on now_ms=%s",
                now_ms,
            )
            return

        logger.info(
            "rest_fallback_poll symbols=%s ws_unhealthy=%s always_on=%s now_ms=%s",
            len(self._config.symbols),
            ws_unhealthy,
            self._config.rest_fallback_always_on,
            now_ms,
        )

        for symbol in self._config.symbols:
            try:
                candles = await loop.run_in_executor(
                    None,
                    lambda sym=symbol: self._fetcher(
                        symbol=sym,
                        interval=self._config.timeframe,
                        limit=self._config.window_candles + 1,
                        rest_base_url=self._config.rest_base_url,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - per-symbol resilience
                logger.warning(
                    "rest_fallback_error symbol=%s reason=fetch_failed err=%s",
                    symbol,
                    exc,
                )
                continue

            if not candles:
                logger.info(
                    "rest_fallback_no_new_closed_candle symbol=%s reason=empty_response",
                    symbol,
                )
                continue

            # Refresh the rolling buffer with everything REST returned (the WS
            # path and the converter both read from this buffer).
            for c in candles:
                self._buffer.upsert(symbol, c)

            latest_closed = latest_closed_candle(candles, now_ms=now_ms)
            if latest_closed is None:
                logger.info(
                    "rest_fallback_no_new_closed_candle symbol=%s reason=no_closed_candle_yet",
                    symbol,
                )
                continue

            logger.info(
                "rest_fallback_latest_closed_candle symbol=%s timeframe=%s candle_open_time=%s "
                "candle_close_time=%s",
                symbol,
                self._config.timeframe,
                ms_to_iso_utc(latest_closed.timestamp),
                ms_to_iso_utc(latest_closed.close_time or latest_closed.timestamp),
            )

            status = await self._stream.publish_closed_candle_for(symbol, latest_closed)
            if status == "published":
                logger.info(
                    "rest_fallback_candidate_publish_allowed symbol=%s candle_open_time=%s",
                    symbol,
                    ms_to_iso_utc(latest_closed.timestamp),
                )
            elif status == "dedup_skipped":
                logger.info(
                    "rest_fallback_candidate_dedup_skipped symbol=%s candle_open_time=%s",
                    symbol,
                    ms_to_iso_utc(latest_closed.timestamp),
                )
            elif status == "not_ready":
                logger.info(
                    "rest_fallback_no_new_closed_candle symbol=%s reason=window_not_ready",
                    symbol,
                )
            else:  # "failed"
                logger.warning(
                    "rest_fallback_error symbol=%s reason=publish_failed candle_open_time=%s",
                    symbol,
                    ms_to_iso_utc(latest_closed.timestamp),
                )
