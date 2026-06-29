"""RabbitMQ broker integration using aio-pika.

Contract
--------
* **Consume** trade candidates from ``data.candidates.ai``.
* **Publish** finalized signals to ``signal.final``.

Reliability
-----------
Every message handler is wrapped in ``try/finally`` so that ``basic_ack`` (or a
``basic_nack``) is *always* sent exactly once, even when inference raises. This
prevents the consumer's unacked window from filling up and stalling the queue
(queue overflow / consumer starvation). Bad messages are dead-lettered (nack
without requeue) unless ``requeue_on_error`` is set.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import orjson

from ae_brain.config import RabbitMQConfig
from ae_brain.contracts import FinalSignal, TradeCandidate
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.amqp")

# A handler maps an inbound candidate to a final signal (or None to skip publish).
SignalHandler = Callable[[TradeCandidate], Awaitable[FinalSignal | None]]


class SignalBroker:
    def __init__(self, cfg: RabbitMQConfig) -> None:
        self._cfg = cfg
        self._connection = None
        self._channel = None
        self._exchange = None

    async def connect(self) -> None:
        import aio_pika

        self._connection = await aio_pika.connect_robust(self._cfg.url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._cfg.prefetch_count)
        if self._cfg.publish_exchange:
            self._exchange = await self._channel.declare_exchange(
                self._cfg.publish_exchange, aio_pika.ExchangeType.TOPIC, durable=True
            )
        else:
            self._exchange = self._channel.default_exchange
        # Ensure queues exist (idempotent).
        await self._channel.declare_queue(self._cfg.consume_queue, durable=True)
        if not self._cfg.publish_exchange:
            await self._channel.declare_queue(self._cfg.publish_routing_key, durable=True)
        log.info("amqp.connected", consume=self._cfg.consume_queue)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            log.info("amqp.closed")

    async def publish_signal(self, signal: FinalSignal) -> None:
        import aio_pika

        body = orjson.dumps(signal.to_dict())
        message = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=signal.correlation_id or None,
        )
        await self._exchange.publish(message, routing_key=self._cfg.publish_routing_key)
        log.info("amqp.published", symbol=signal.symbol, decision=signal.decision.value)

    async def consume(self, handler: SignalHandler) -> None:
        """Start consuming candidates; runs until cancelled."""
        import aio_pika

        queue = await self._channel.declare_queue(self._cfg.consume_queue, durable=True)

        async def _on_message(message: "aio_pika.abc.AbstractIncomingMessage") -> None:
            acked = False
            try:
                payload = orjson.loads(message.body)
                candidate = TradeCandidate.from_message(payload)
                signal = await handler(candidate)
                if signal is not None:
                    await self.publish_signal(signal)
                await message.ack()
                acked = True
            except orjson.JSONDecodeError as exc:
                log.error("amqp.bad_payload", err=str(exc))
                await message.nack(requeue=False)  # poison message -> dead-letter
                acked = True
            except Exception as exc:  # noqa: BLE001 - never let one msg kill the loop
                log.exception("amqp.handler_error", err=str(exc))
                await message.nack(requeue=self._cfg.requeue_on_error)
                acked = True
            finally:
                # Guarantee the broker is never left waiting on an unacked msg.
                if not acked:
                    try:
                        await message.nack(requeue=self._cfg.requeue_on_error)
                    except Exception:  # pragma: no cover
                        log.error("amqp.ack_failed", delivery_tag=message.delivery_tag)

        await queue.consume(_on_message, consumer_tag=self._cfg.consumer_tag)
        log.info("amqp.consuming", queue=self._cfg.consume_queue)
        # Block forever (until task cancellation).
        await asyncio.Future()
