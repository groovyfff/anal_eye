"""Shared async-friendly RabbitMQ publisher (declare exchange only — no consumer queues)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pika

from shared.utils.rabbitmq_topology import EXCHANGE, declare_exchange


class PikaClient:
    def __init__(self, url: str, default_exchange: str = EXCHANGE) -> None:
        self.url = url
        self.default_exchange = default_exchange
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pika-client")
        self._executor_thread_id: int | None = None

    async def connect(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._connect_blocking)
            return True
        except Exception:
            return False

    def _connect_blocking(self) -> None:
        self._executor_thread_id = threading.get_ident()
        if self._connection and self._connection.is_open:
            with contextlib.suppress(Exception):
                self._connection.close()
        params = pika.URLParameters(self.url)
        self._connection = pika.BlockingConnection(parameters=params)
        self._channel = self._connection.channel()
        declare_exchange(self._channel, self.default_exchange)

    def _ensure_connected_blocking(self) -> None:
        if self._connection and self._connection.is_open and self._channel is not None:
            return
        self._connect_blocking()

    def publish(self, exchange_name: str, routing_key: str, body: str) -> None:
        if self._executor_thread_id == threading.get_ident():
            self._publish_blocking(exchange_name, routing_key, body)
            return
        future = self._executor.submit(self._publish_blocking, exchange_name, routing_key, body)
        future.result()

    async def publish_async(self, exchange_name: str, routing_key: str, body: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._publish_blocking, exchange_name, routing_key, body)

    def _publish_blocking(self, exchange_name: str, routing_key: str, body: str) -> None:
        payload = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        for attempt in range(1, 3):
            try:
                self._ensure_connected_blocking()
                if self._channel is None:
                    raise RuntimeError("RabbitMQ client is not connected")
                self._channel.basic_publish(
                    exchange=exchange_name,
                    routing_key=routing_key,
                    body=payload.encode("utf-8"),
                    properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
                )
                return
            except Exception:
                self._connection = None
                self._channel = None
                if attempt == 2:
                    raise

    async def close(self) -> None:
        if self._connection and self._connection.is_open:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._connection.close)
        self._executor.shutdown(wait=False, cancel_futures=False)
