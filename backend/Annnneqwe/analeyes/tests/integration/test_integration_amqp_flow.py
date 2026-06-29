"""Layer 2 - real AMQP round-trip: data.candidates.ai -> ae_brain -> signal.final.

Boots real PostgreSQL + RabbitMQ containers, pre-INSERTs a candidate row, then
drives the actual ``ae_brain.runtime.LiveRuntime`` wiring (DB + InferenceEngine +
SignalBroker). A backend-shaped candidate (64-candle window, stock asset class,
valid ``signal_log_db_id``) is published to ``data.candidates.ai``; we assert the
consumer:

* processes it (no nack / dead-letter),
* UPDATEs the pre-inserted row in the real schema, and
* publishes a ``signal.final`` carrying the exact tracker-service keys.
"""

from __future__ import annotations

import asyncio
import contextlib

import aio_pika
import orjson
import pytest
from sqlalchemy import select

from ae_brain.config import Settings
from ae_brain.runtime import LiveRuntime
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from shared.database.signal_log_repository import save_external_candidate_log

CANDIDATE_RK = "data.candidates.ai"
SIGNAL_FINAL_RK = "signal.final"


async def _await_message(channel: aio_pika.abc.AbstractChannel, queue_name: str, timeout: float = 40.0):
    """Poll a queue for a single message until timeout (None if none arrived)."""
    queue = await channel.declare_queue(queue_name, durable=True)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        msg = await queue.get(no_ack=True, fail=False)
        if msg is not None:
            return msg
        await asyncio.sleep(0.25)
    return None


def _make_settings(pg_params: dict, amqp_url: str) -> Settings:
    settings = Settings()
    settings.executor.process_workers = 0  # in-process thread pool only
    settings.rabbitmq.url = amqp_url
    settings.rabbitmq.publish_exchange = ""  # default exchange => queue == routing_key
    settings.rabbitmq.consume_queue = CANDIDATE_RK
    settings.rabbitmq.publish_routing_key = SIGNAL_FINAL_RK
    settings.database.host = pg_params["host"]
    settings.database.port = pg_params["port"]
    settings.database.user = pg_params["user"]
    settings.database.password = pg_params["password"]
    settings.database.name = pg_params["name"]
    return settings


def test_amqp_candidate_to_signal_final(pg_params, amqp_url, candidate_factory):
    async def run() -> None:
        # Schema + backend pre-INSERT.
        await DatabaseManager.initialize(database_url=pg_params["async_url"])
        payload = candidate_factory(asset_class="stock", n_candles=64)
        db_id: int | None = None
        async for session in DatabaseManager.get_session():
            db_id = await save_external_candidate_log(session, payload)
        assert db_id and db_id > 0
        payload["signal_log_db_id"] = db_id
        # ae_brain runtime owns its own asyncpg pool to the same DB.
        await DatabaseManager.close()

        runtime = LiveRuntime(_make_settings(pg_params, amqp_url))
        await runtime._db.connect()
        runtime._engine.load_models()
        await runtime._broker.connect()
        consume_task = asyncio.create_task(runtime._broker.consume(runtime._handle))

        conn = await aio_pika.connect_robust(amqp_url)
        channel = await conn.channel()
        try:
            await channel.default_exchange.publish(
                aio_pika.Message(body=orjson.dumps(payload), content_type="application/json"),
                routing_key=CANDIDATE_RK,
            )

            final_msg = await _await_message(channel, SIGNAL_FINAL_RK, timeout=40.0)
            assert final_msg is not None, "no signal.final published (lost or dead-lettered)"
            data = orjson.loads(final_msg.body)

            # Exact tracker-service contract keys must be present.
            for key in ("tp", "sl", "signal_id", "source_ai", "decision", "asset_class", "signal_log_db_id"):
                assert key in data, f"missing tracker key: {key}"
            assert data["source_ai"] == "ensemble"
            assert data["asset_class"] == "stock"
            assert data["signal_id"] == payload["signal_id"]
            assert data["signal_log_db_id"] == db_id

            # The consumer performed the strict UPDATE on the pre-inserted row.
            await DatabaseManager.initialize(database_url=pg_params["async_url"])
            try:
                async for session in DatabaseManager.get_session():
                    row = (
                        await session.execute(
                            select(SignalFeatureLog).where(SignalFeatureLog.id == db_id)
                        )
                    ).scalar_one()
                    assert row.ai_signal_type is not None  # row was updated by ae_brain
                    assert row.ai_signal_type == data["decision"]
            finally:
                await DatabaseManager.close()
        finally:
            consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consume_task
            await conn.close()
            await runtime.shutdown()

    asyncio.run(run())
