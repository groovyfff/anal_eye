"""RabbitMQ consumer for news sentiment snapshots.

This is the **only** integration between the news service and the AE Brain
mathematical model. It consumes ``data.news.sentiment`` snapshots published by
``news-sentiment-service`` and caches the latest snapshot per symbol in memory.

The model never calls the news service directly; it reads from this cache.
If no fresh snapshot exists for a symbol, scoring continues without news
features (graceful degradation). The consumer is **optional** (gated by
``Settings.enable_news_features``) so existing deployments are unaffected.

Boundary contract: this module imports only ``aio_pika`` and stdlib — it never
imports news-service code, and nothing in news-service imports this.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from ae_brain.messaging.amqp_utils import parse_amqp_url
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.news_features")


class NewsFeaturesCache:
    """Thread-safe-ish (asyncio) cache of the latest snapshot per symbol.

    A snapshot is dropped once it is older than ``max_age_s`` (freshness TTL),
    so stale news never influences a candidate.
    """

    def __init__(self, max_age_s: float = 300.0) -> None:
        self._max_age_s = max(0.0, max_age_s)
        # symbol -> (monotonic_ts, snapshot_dict)
        self._store: Dict[str, tuple[float, Dict[str, Any]]] = {}

    def update(self, symbol: str, snapshot: Dict[str, Any]) -> None:
        if not symbol or not isinstance(snapshot, dict):
            return
        self._store[str(symbol).upper()] = (time.monotonic(), snapshot)

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the fresh snapshot for ``symbol`` or None if absent/stale."""
        key = str(symbol or "").upper()
        if not key:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, snapshot = entry
        if self._max_age_s > 0 and (time.monotonic() - ts) > self._max_age_s:
            # Expired — drop and report absent.
            self._store.pop(key, None)
            return None
        return snapshot

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)


class NewsFeaturesConsumer:
    """Consumes ``data.news.sentiment`` and feeds :class:`NewsFeaturesCache`.

    Mirrors the connect/consume pattern of :class:`SignalBroker` but is fully
    isolated: it owns its own aio-pika connection/channel/queue and only writes
    into the cache. It never touches candidate processing or the engine.
    """

    QUEUE = "q_data_news_sentiment"
    ROUTING_KEY = "data.news.sentiment"
    EXCHANGE = "analeyes.events"

    def __init__(
        self,
        amqp_url: str,
        cache: NewsFeaturesCache,
        *,
        exchange: str = EXCHANGE,
        queue: str = QUEUE,
        routing_key: str = ROUTING_KEY,
        consumer_tag: str = "ae-brain-news-features",
        requeue_on_error: bool = False,
    ) -> None:
        self._url = amqp_url
        self._cache = cache
        self._exchange = exchange
        self._queue = queue
        self._routing_key = routing_key
        self._consumer_tag = consumer_tag
        self._requeue_on_error = requeue_on_error
        self._connection = None
        self._channel = None
        self._consume_task: Optional[asyncio.Task] = None
        endpoint = parse_amqp_url(amqp_url)
        # Validate it's an amqp(s) url; if not, connect() will refuse.
        self._endpoint = endpoint

    async def connect(self) -> None:
        import aio_pika
        import orjson

        self._connection = await aio_pika.connect_robust(self._endpoint.url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=16)
        exchange = await self._channel.declare_exchange(
            self._exchange, aio_pika.ExchangeType.TOPIC, durable=True
        )
        queue = await self._channel.declare_queue(self._queue, durable=True)
        await queue.bind(exchange, routing_key=self._routing_key)
        log.info(
            "news_features.consumer_connected",
            exchange=self._exchange,
            queue=self._queue,
            routing_key=self._routing_key,
            host=self._endpoint.host,
            vhost=self._endpoint.vhost,
        )

    async def consume(self) -> None:
        import aio_pika
        import orjson

        if self._channel is None:
            raise RuntimeError("NewsFeaturesConsumer.consume called before connect()")

        queue = await self._channel.declare_queue(self._queue, durable=True)

        async def _on_message(message: "aio_pika.abc.AbstractIncomingMessage") -> None:
            acked = False
            symbol = ""
            try:
                try:
                    payload = orjson.loads(message.body)
                except orjson.JSONDecodeError:
                    log.warning(
                        "news_features.invalid_json",
                        routing_key=message.routing_key,
                        ack="skipped_and_acked",
                    )
                    await message.ack()
                    acked = True
                    return
                if not isinstance(payload, dict):
                    log.warning("news_features.non_dict_payload", ack="skipped_and_acked")
                    await message.ack()
                    acked = True
                    return
                symbol = str(payload.get("symbol", "")).upper()
                if not symbol:
                    log.info("news_features.skip_no_symbol", ack="skipped_and_acked")
                    await message.ack()
                    acked = True
                    return
                self._cache.update(symbol, payload)
                log.info(
                    "news_features.cached",
                    symbol=symbol,
                    sentiment=payload.get("news_sentiment"),
                    volume=payload.get("news_volume"),
                    ack="acked",
                )
                await message.ack()
                acked = True
            except Exception as exc:  # noqa: BLE001 - never poison-loop
                log.exception("news_features.handler_error", symbol=symbol, err=str(exc))
                if not acked:
                    await message.nack(requeue=self._requeue_on_error)
            finally:
                if not acked:
                    try:
                        await message.nack(requeue=self._requeue_on_error)
                    except Exception:
                        log.error("news_features.ack_failed", delivery_tag=message.delivery_tag)

        await queue.consume(_on_message, consumer_tag=self._consumer_tag)
        log.info(
            "news_features.consumer_registered",
            queue=self._queue,
            routing_key=self._routing_key,
        )
        await asyncio.Future()  # run until cancelled

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:  # noqa: BLE001
                pass
        log.info("news_features.closed")


def attach_news_features_to_candidate(
    cache: NewsFeaturesCache,
    candidate_meta: Dict[str, Any],
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """Attach the latest fresh news snapshot to a candidate's meta (echo-only).

    Writes the snapshot under ``candidate_meta["news"]`` and mirrors a few
    scalar fields under ``candidate_meta["features"]`` with ``news_`` prefixes
    so they appear in the published ``signal.final`` echo. Returns the snapshot
    or None when no fresh news exists (in which case the candidate is untouched
    and scoring proceeds normally).

    NOTE: AE Brain re-derives scoring features from candles, so this does NOT
    change trade decisions yet — it only makes news features available for
    logging/echo and for a future fusion-layer wiring.
    """
    snapshot = cache.get(symbol)
    if snapshot is None:
        return None
    candidate_meta["news"] = snapshot
    features = candidate_meta.setdefault("features", {})
    features["news_sentiment"] = snapshot.get("news_sentiment", 0.0)
    features["news_volume"] = snapshot.get("news_volume", 0.0)
    features["news_bullish_count"] = snapshot.get("bullish_count", 0)
    features["news_bearish_count"] = snapshot.get("bearish_count", 0)
    features["news_manipulation_risk_avg"] = snapshot.get("manipulation_risk_avg", 0.0)
    features["news_recency_s"] = snapshot.get("news_recency_s", 0)
    return snapshot
