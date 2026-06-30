"""Central RabbitMQ exchange, queue, and routing-key definitions for AnalEyes V6."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pika.adapters.blocking_connection

# Topic exchange used by all application services.
EXCHANGE = "analeyes.events"
EXCHANGE_TYPE = "topic"


class RoutingKey:
    DATA_RAW_BINANCE = "data.raw.binance"
    DATA_CANDIDATES_AI = "data.candidates.ai"
    DATA_LIVE_PRICES_EXTERNAL = "data.live_prices.external"
    SIGNAL_FINAL = "signal.final"
    SIGNAL_OUTCOME = "signal.outcome"
    SIGNAL_ENTRY_EVENT = "signal.entry_event"
    STATUS_NOTIFICATION = "status.notification"
    STATUS_TRACKER = "status.tracker"
    STATUS_COLLECTOR = "status.collector"
    STATUS_ALL = "status.*"
    CONTROL_BACKTEST_START = "control.backtest.start"
    AI_BACKTEST_ANALYZE = "ai.backtest.analyze"


class Queue:
    DATA_RAW_BINANCE = "q_data_raw_binance"
    DATA_CANDIDATES_AI = "q_data_candidates_ai"
    NEW_SIGNALS = "q_new_signals"
    SIGNAL_OUTCOMES = "q_signal_outcomes"
    SIGNAL_ENTRY_EVENTS = "q_signal_entry_events"
    TRACKER_SIGNALS = "q_tracker_signals"
    LIVE_PRICES_EXTERNAL = "q_data_live_prices_external"
    API_STATUS = "q_api_status"
    CONTROL_BACKTEST_START = "q_control_backtest_start"
    AI_BACKTEST_ANALYZE = "q_ai_backtest_analyze"


# (queue_name, routing_key) — each consumer gets its own queue.
QUEUE_BINDINGS: list[tuple[str, str]] = [
    (Queue.DATA_RAW_BINANCE, RoutingKey.DATA_RAW_BINANCE),
    (Queue.DATA_CANDIDATES_AI, RoutingKey.DATA_CANDIDATES_AI),
    (Queue.NEW_SIGNALS, RoutingKey.SIGNAL_FINAL),
    (Queue.TRACKER_SIGNALS, RoutingKey.SIGNAL_FINAL),
    (Queue.SIGNAL_OUTCOMES, RoutingKey.SIGNAL_OUTCOME),
    (Queue.SIGNAL_ENTRY_EVENTS, RoutingKey.SIGNAL_ENTRY_EVENT),
    (Queue.LIVE_PRICES_EXTERNAL, RoutingKey.DATA_LIVE_PRICES_EXTERNAL),
    (Queue.API_STATUS, RoutingKey.STATUS_ALL),
    (Queue.CONTROL_BACKTEST_START, RoutingKey.CONTROL_BACKTEST_START),
    (Queue.AI_BACKTEST_ANALYZE, RoutingKey.AI_BACKTEST_ANALYZE),
]


def declare_exchange(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    exchange: str = EXCHANGE,
) -> None:
    channel.exchange_declare(exchange=exchange, exchange_type=EXCHANGE_TYPE, durable=True)


def declare_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    *,
    passive: bool = False,
) -> None:
    channel.queue_declare(queue=queue, durable=True, passive=passive)


def bind_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    routing_key: str,
    exchange: str = EXCHANGE,
) -> None:
    channel.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)


def ensure_queue_binding(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    queue: str,
    routing_key: str,
    exchange: str = EXCHANGE,
    *,
    passive_first: bool = True,
) -> None:
    """Passive-check queue, declare if missing, then bind."""
    import pika

    declare_exchange(channel, exchange)
    if passive_first:
        try:
            channel.queue_declare(queue=queue, durable=True, passive=True)
        except pika.exceptions.AMQPChannelError:
            channel = channel.connection.channel()
            declare_exchange(channel, exchange)
            channel.queue_declare(queue=queue, durable=True)
    else:
        channel.queue_declare(queue=queue, durable=True)
    bind_queue(channel, queue, routing_key, exchange)
