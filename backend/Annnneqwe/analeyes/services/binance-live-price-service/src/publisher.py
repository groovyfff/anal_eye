from __future__ import annotations

import json
import logging
from typing import Any

from shared.utils.pika_client import PikaClient
from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

logger = logging.getLogger(__name__)


class LivePricePublisher:
    def __init__(self, url: str, *, user: str, vhost: str) -> None:
        self._user = user
        self._vhost = vhost
        self._client = PikaClient(url, default_exchange=EXCHANGE)
        self._connected = False

    async def connect(self) -> None:
        ok = await self._client.connect()
        if not ok:
            raise RuntimeError("RabbitMQ connection failed")
        self._connected = True
        logger.info(
            "RabbitMQ connected user=%s vhost=%s exchange=%s",
            self._user,
            self._vhost,
            EXCHANGE,
        )

    async def reconnect(self) -> None:
        self._connected = False
        await self.connect()

    async def publish_live_price(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False)
        try:
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_LIVE_PRICES_EXTERNAL, body)
        except Exception:
            logger.error(
                "RabbitMQ publish failed rk=%s symbol=%s — reconnecting",
                RoutingKey.DATA_LIVE_PRICES_EXTERNAL,
                payload.get("symbol"),
            )
            await self.reconnect()
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_LIVE_PRICES_EXTERNAL, body)

    async def publish_raw_kline(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False)
        try:
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_RAW_BINANCE, body)
        except Exception:
            logger.error(
                "RabbitMQ publish failed rk=%s symbol=%s — reconnecting",
                RoutingKey.DATA_RAW_BINANCE,
                payload.get("symbol"),
            )
            await self.reconnect()
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_RAW_BINANCE, body)

    async def close(self) -> None:
        await self._client.close()
