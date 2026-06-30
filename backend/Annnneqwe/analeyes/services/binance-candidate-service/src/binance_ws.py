from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from src.candle_buffer import Candle, CandleBuffer
from src.candidate_builder import build_candidate_payload
from src.config import ServiceConfig
from src.kline_parser import parse_kline_message
from src.publish_policy import PublishPolicy
from src.publisher import CandidatePublisher

logger = logging.getLogger(__name__)


class BinanceCandidateStream:
    def __init__(
        self,
        config: ServiceConfig,
        buffer: CandleBuffer,
        publisher: CandidatePublisher,
        policy: PublishPolicy,
    ) -> None:
        self._config = config
        self._buffer = buffer
        self._publisher = publisher
        self._policy = policy

    async def run_forever(self) -> None:
        delay = self._config.reconnect_delay_sec
        while True:
            wss_url = self._config.build_wss_url()
            stream_label = ",".join(self._config.all_stream_names())
            try:
                async with websockets.connect(
                    wss_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=2**20,
                ) as ws:
                    logger.info("Binance WSS connected stream=%s", stream_label)
                    async for raw_message in ws:
                        await self._handle_message(raw_message, stream_label)
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                reason = f"code={exc.code} reason={exc.reason or 'connection closed'}"
                logger.warning("Binance WSS disconnected; reconnecting in %ss reason=%s", delay, reason)
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__
                logger.warning("Binance WSS disconnected; reconnecting in %ss reason=%s", delay, reason)
            await asyncio.sleep(delay)

    async def _handle_message(self, raw_message: str | bytes, stream_label: str) -> None:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            message: dict[str, Any] = json.loads(raw_message)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Invalid Binance kline reason=invalid json: %s", exc)
            return

        default_stream = self._config.all_stream_names()[0] if len(self._config.symbols) == 1 else None
        try:
            parsed = parse_kline_message(message, timeframe=self._config.timeframe, default_stream=default_stream)
        except ValueError as exc:
            logger.warning("Invalid Binance kline reason=%s", exc)
            return

        candle = Candle(
            timestamp=parsed.open_time,
            open=parsed.open,
            high=parsed.high,
            low=parsed.low,
            close=parsed.close,
            volume=parsed.volume,
            closed=parsed.is_closed,
        )
        self._buffer.upsert(parsed.symbol, candle)
        await self._maybe_publish(parsed.symbol, parsed.event_time, parsed.is_closed)

    async def _maybe_publish(self, symbol: str, event_time: int, candle_closed: bool) -> None:
        count = self._buffer.count(symbol)
        min_candles = self._config.min_candles
        if count < min_candles:
            logger.info(
                "Skipping candidate publish symbol=%s reason=not_enough_candles count=%s min=%s",
                symbol,
                count,
                min_candles,
            )
            return

        if not self._policy.should_publish(symbol, candle_closed=candle_closed):
            return

        candles = self._buffer.candles(symbol)
        payload = build_candidate_payload(
            symbol=symbol,
            market=self._config.market,
            timeframe=self._config.timeframe,
            event_time=event_time,
            candles=candles,
        )
        await self._publisher.publish_candidate(payload)
        self._policy.mark_published(symbol)
        logger.info(
            "Published Binance candidate symbol=%s candles=%s price=%s composite=%s rk=data.candidates.ai",
            symbol,
            len(candles),
            payload["current_price"],
            payload["composite_score"],
        )
