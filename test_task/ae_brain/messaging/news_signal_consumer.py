"""RabbitMQ consumer for ``news.market_signal`` messages.

This consumes the per-item market signals produced by the news-service's
OpenRouter path and stores them in a :class:`NewsContextStore` for the fusion
layer to read. It is fully isolated (own aio-pika connection/channel/queue) and
mirrors :class:`NewsFeaturesConsumer` exactly — the only consumer pattern used
elsewhere in AE Brain.

Boundary contract: imports only ``aio_pika`` + stdlib + the local store. Never
imports news-service code. If the queue is missing/broken or a message is
malformed, it is logged and ACKed (no poison loop); AE Brain's candidate loop is
never affected.

Diagnostic logs:
* ``news_consumer_registered`` — consumer bound the queue.
* ``news_signal_received``     — a message arrived.
* ``news_signal_cached``       — a signal was stored (emitted by the store).
* ``news_signal_rejected``     — a signal/message was invalid (store or here).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ae_brain.messaging.amqp_utils import parse_amqp_url
from ae_brain.messaging.news_context_store import NewsContextStore
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.news_signal_consumer")


class NewsSignalConsumer:
    """Consumes ``news.market_signal`` and feeds a :class:`NewsContextStore`."""

    QUEUE = "q_news_market_signal"
    ROUTING_KEY = "news.market_signal"
    EXCHANGE = "analeyes.events"

    def __init__(
        self,
        amqp_url: str,
        store: NewsContextStore,
        *,
        exchange: str = EXCHANGE,
        queue: str = QUEUE,
        routing_key: str = ROUTING_KEY,
        min_relevance: float = 0.65,
        consumer_tag: str = "ae-brain-news-market-signal",
        requeue_on_error: bool = False,
    ) -> None:
        self._url = amqp_url
        self._store = store
        self._exchange = exchange
        self._queue = queue
        self._routing_key = routing_key
        self._min_relevance = float(min_relevance)
        self._consumer_tag = consumer_tag
        self._requeue_on_error = requeue_on_error
        self._connection = None
        self._channel = None
        self._endpoint = parse_amqp_url(amqp_url)

    async def connect(self) -> None:
        import aio_pika

        self._connection = await aio_pika.connect_robust(self._endpoint.url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=16)
        exchange = await self._channel.declare_exchange(
            self._exchange, aio_pika.ExchangeType.TOPIC, durable=True
        )
        queue = await self._channel.declare_queue(self._queue, durable=True)
        await queue.bind(exchange, routing_key=self._routing_key)
        log.info(
            "news_consumer_registered",
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
            raise RuntimeError("NewsSignalConsumer.consume called before connect()")

        queue = await self._channel.declare_queue(self._queue, durable=True)

        async def _on_message(message: "aio_pika.abc.AbstractIncomingMessage") -> None:
            acked = False
            news_id = ""
            try:
                try:
                    payload = orjson.loads(message.body)
                except orjson.JSONDecodeError:
                    log.warning("news_signal_rejected", reason="invalid_json", ack="acked")
                    await message.ack()
                    acked = True
                    return
                if not isinstance(payload, dict):
                    log.warning("news_signal_rejected", reason="non_dict_payload", ack="acked")
                    await message.ack()
                    acked = True
                    return
                news_id = str(payload.get("news_id", ""))
                signals = payload.get("signals")
                if not isinstance(signals, list) or not signals:
                    log.info("news_signal_rejected", news_id=news_id, reason="no_signals", ack="acked")
                    await message.ack()
                    acked = True
                    return
                log.info("news_signal_received", news_id=news_id,
                         signal_count=len(signals))
                for sig in signals:
                    if not isinstance(sig, dict):
                        continue
                    symbol = str(sig.get("symbol", "")).upper()
                    # Final relevance gate (defensive; news-service already filters).
                    try:
                        rel = float(sig.get("relevance", 0.0))
                    except (TypeError, ValueError):
                        rel = 0.0
                    if rel < self._min_relevance:
                        log.info("news_signal_rejected", symbol=symbol,
                                 reason="below_min_relevance", relevance=rel)
                        continue
                    self._store.add_signal_dict(symbol, sig)
                await message.ack()
                acked = True
            except Exception as exc:  # noqa: BLE001 - never poison-loop
                log.exception("news_signal.handler_error", news_id=news_id, err=str(exc))
                if not acked:
                    await message.nack(requeue=self._requeue_on_error)
            finally:
                if not acked:
                    try:
                        await message.nack(requeue=self._requeue_on_error)
                    except Exception:
                        log.error("news_signal.ack_failed", delivery_tag=message.delivery_tag)

        await queue.consume(_on_message, consumer_tag=self._consumer_tag)
        await asyncio.Future()  # run until cancelled

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:  # noqa: BLE001
                pass
        log.info("news_signal_consumer.closed")
