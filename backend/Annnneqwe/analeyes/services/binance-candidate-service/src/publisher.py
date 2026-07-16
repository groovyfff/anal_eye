from __future__ import annotations

import json
import logging
from typing import Any

from shared.symbol_universe import default_allowed_symbols, is_symbol_allowed
from shared.utils.pika_client import PikaClient
from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

from src.converters.ae_brain_candidate import validate_candidate_payload

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
            "RabbitMQ connected user=%s vhost=%s exchange=%s routing_key=%s source=binance_futures_api",
            self._user,
            self._vhost,
            EXCHANGE,
            RoutingKey.DATA_CANDIDATES_AI,
        )

    async def reconnect(self) -> None:
        await self.connect()

    async def publish_candidate(self, payload: dict[str, Any]) -> None:
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            logger.error("candidate_publish_failed reason=missing_symbol keys=%s", list(payload.keys()))
            raise ValueError("missing_symbol")
        allowed = default_allowed_symbols()
        if not is_symbol_allowed(symbol, allowed):
            logger.info(
                "candidate_rejected_symbol symbol=%s allowed=%s",
                symbol,
                ",".join(sorted(allowed)),
            )
            raise ValueError(f"symbol_not_allowed:{symbol}")
        payload["symbol"] = symbol
        try:
            validate_candidate_payload(payload)
        except ValueError as exc:
            logger.info("candidate_publish_failed symbol=%s reason=%s", symbol, exc)
            raise
        body = json.dumps(payload, ensure_ascii=False)
        try:
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_CANDIDATES_AI, body)
        except Exception as exc:
            logger.error(
                "candidate_publish_failed symbol=%s rk=%s err=%s — reconnecting",
                symbol,
                RoutingKey.DATA_CANDIDATES_AI,
                exc,
            )
            await self.reconnect()
            await self._client.publish_async(EXCHANGE, RoutingKey.DATA_CANDIDATES_AI, body)
        logger.info(
            "candidate_publish_allowed symbol=%s routing_key=%s exchange=%s",
            symbol,
            RoutingKey.DATA_CANDIDATES_AI,
            EXCHANGE,
        )

    async def close(self) -> None:
        await self._client.close()
