"""Peek at one candidate message without consuming it."""

from __future__ import annotations

import asyncio
import json
import sys

from ae_brain.config import get_settings
from ae_brain.messaging.amqp_utils import log_endpoint, parse_amqp_url
from ae_brain.messaging.candidate_normalizer import normalize_candidate
from ae_brain.utils.logging import configure_logging, get_logger

log = get_logger("ae_brain.tools.dump_one_candidate")


async def _run() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = settings.amqp_input
    endpoint = parse_amqp_url(cfg.resolved_url)

    import aio_pika

    connection = await aio_pika.connect_robust(endpoint.url)
    channel = await connection.channel()
    exchange = await channel.declare_exchange(cfg.exchange, aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue(cfg.queue, durable=True)
    await queue.bind(exchange, routing_key=cfg.routing_key)

    print(log_endpoint("input", endpoint, exchange=cfg.exchange, queue=cfg.queue, routing_key=cfg.routing_key))

    message = await queue.get(no_ack=False, fail=False)
    if message is None:
        print("No message available on queue", cfg.queue)
        await connection.close()
        return 1

    try:
        payload = json.loads(message.body)
    except json.JSONDecodeError:
        print("invalid_json body_size=", len(message.body))
        await message.nack(requeue=True)
        await connection.close()
        return 1

    keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    print(f"delivery_tag={message.delivery_tag} routing_key={message.routing_key} body_size={len(message.body)}")
    print("raw top-level keys:", keys)

    norm = normalize_candidate(payload, min_composite_score=settings.min_composite_score)
    print("normalized summary:", norm.summary)
    print("skip_reason:", norm.skip_reason)
    print("direction_hint:", norm.direction_hint)
    if norm.payload:
        print(json.dumps(norm.payload, indent=2, default=str)[:4000])

    await message.nack(requeue=True)
    await connection.close()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
