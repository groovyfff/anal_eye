#!/usr/bin/env python3
"""RabbitMQ connectivity + signal.final queue probe (safe: basic_nack requeue)."""

from __future__ import annotations

import socket
import sys

import pika

from shared.rabbitmq_config import rabbitmq_connection_info, resolve_rabbitmq_url, sanitized_rabbitmq_url
from shared.utils.rabbitmq_topology import Queue


def main() -> int:
    queue = Queue.NEW_SIGNALS
    try:
        url = resolve_rabbitmq_url()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    info = rabbitmq_connection_info(url)
    print("=== RabbitMQ debug ===")
    print(f"url_sanitized: {info['url_sanitized']}")
    print(f"host: {info['host']}")
    print(f"port: {info['port']}")
    print(f"user: {info['user']}")
    print(f"vhost: {info['vhost']}")

    if info["host"]:
        try:
            ip = socket.gethostbyname(info["host"])
            print(f"resolved_ip: {ip}")
        except socket.gaierror as exc:
            print(f"resolved_ip: DNS failed ({exc})")

    params = pika.URLParameters(url)
    params.heartbeat = 60
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    try:
        result = channel.queue_declare(queue=queue, passive=True)
        print(f"queue: {queue}")
        print(f"message_count: {result.method.message_count}")
        print(f"consumer_count: {result.method.consumer_count}")

        method, properties, body = channel.basic_get(queue=queue, auto_ack=False)
        if method is None:
            print("basic_get: no message available (queue empty or all unacked)")
        else:
            preview = body[:200]
            print(f"basic_get: delivery_tag={method.delivery_tag} exchange={method.exchange!r} rk={method.routing_key!r}")
            print(f"basic_get body preview: {preview!r}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            print("basic_nack: requeued=True (message preserved)")
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()

    print(f"connected_as: {info['user']} vhost={info['vhost']}")
    print(f"full_url_check: {sanitized_rabbitmq_url(url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
