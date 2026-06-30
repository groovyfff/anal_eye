from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from src.config import ServiceConfig
from src.kline_parser import parse_kline_message
from src.publisher import LivePricePublisher

logger = logging.getLogger(__name__)


class BinanceKlineStream:
    def __init__(self, config: ServiceConfig, publisher: LivePricePublisher) -> None:
        self._config = config
        self._publisher = publisher
        self._published_count = 0

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
            logger.warning("Invalid Binance kline message reason=invalid json: %s", exc)
            return

        default_stream = self._config.all_stream_names()[0] if len(self._config.symbols) == 1 else None
        try:
            live_price, raw_kline = parse_kline_message(
                message,
                market=self._config.market,
                timeframe=self._config.timeframe,
                default_stream=default_stream,
            )
        except ValueError as exc:
            logger.warning("Invalid Binance kline message reason=%s", exc)
            return

        await self._publisher.publish_live_price(live_price)
        if self._config.publish_raw_kline and raw_kline is not None:
            await self._publisher.publish_raw_kline(raw_kline)

        self._published_count += 1
        if self._published_count == 1 or self._published_count % self._config.log_every_n == 0:
            logger.info(
                "Published Binance live price symbol=%s price=%s ts=%s closed=%s rk=data.live_prices.external",
                live_price["symbol"],
                live_price["price"],
                live_price["ts"],
                live_price["is_candle_closed"],
            )
