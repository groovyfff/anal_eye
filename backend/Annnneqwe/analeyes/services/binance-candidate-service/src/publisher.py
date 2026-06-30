from __future__ import annotations

import json
import logging
from typing import Any

from shared.utils.pika_client import PikaClient
from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

logger = logging.getLogger(__name__)


class CandidatePublisher:
    def __init__(self, url: str, *, user: str, vhost: str) -> None:
        self._user = user
        self._vhost = vhost
        self._client = PikaClient(url, default_exchange=EXCHANGE)

    async def connect(self) -> None:
        ok = await self._client.connect()
        if not ok:
            raise RuntimeError("RabbitMQ connection failed")
        logger.info(
            "RabbitMQ connected user=%s vhost=%s exchange=%s",
            self._user,
            self._vhost,
            EXCHANGE,
        )

    async def reconnect(self) -> None:
        await self.connect()

    async def publish_candidate(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False)
        try:
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_CANDIDATES_AI, body)
        except Exception:
            logger.error(
                "RabbitMQ publish failed rk=%s symbol=%s — reconnecting",
                RoutingKey.DATA_CANDIDATES_AI,
                payload.get("symbol"),
            )
            await self.reconnect()
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_CANDIDATES_AI, body)

    async def close(self) -> None:
        await self._client.close()
