#!/usr/bin/env python3
"""Smoke-test symbol safety: publish sample candidates and verify gate behavior.

Usage (from test_task/ with RabbitMQ running):

  PYTHONPATH=. .venv/bin/python scripts/smoke_symbol_safety.py

Publishes four synthetic candidates to ``data.candidates.ai`` and prints queue
counts plus expected outcomes. Run ae-brain and notification-service alongside
to observe end-to-end filtering in logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import aio_pika

EXCHANGE = "analeyes.events"
CANDIDATE_RK = "data.candidates.ai"
SIGNAL_FINAL_RK = "signal.final"
CANDIDATE_QUEUE = "q_data_candidates_ai"
SIGNAL_QUEUE = "q_new_signals"

ALLOWED = frozenset(
    {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"}
)


def _base_candidate(symbol: str, *, confidence_hint: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "asset_class": "crypto",
        "interval": "1h",
        "signal_log_db_id": 0,
        "composite_score": 0.85,
        "current_price": 100.0,
        "candles": [
            {
                "open_time": 1_700_000_000_000 + i * 3_600_000,
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
            }
            for i in range(100)
        ],
    }
    if confidence_hint is not None:
        payload["_smoke_confidence_hint"] = confidence_hint
    return payload


SCENARIOS = [
    ("BTCUSDT allowed", _base_candidate("BTCUSDT"), "upstream+ae-brain may process"),
    ("ETHUSDT allowed", _base_candidate("ETHUSDT"), "upstream+ae-brain may process"),
    ("ADAUSDT rejected", _base_candidate("ADAUSDT"), "rejected: unsupported_symbol"),
    ("low confidence hint", _base_candidate("BTCUSDT", confidence_hint=0.395), "ae-brain suppresses if decision LONG <0.70"),
]


async def _queue_depth(connection: aio_pika.abc.AbstractConnection, queue_name: str) -> int:
    channel = await connection.channel()
    try:
        result = await channel.declare_queue(queue_name, durable=True, passive=True)
        return result.declaration_result.message_count
    except Exception:
        return -1
    finally:
        await channel.close()


async def _publish(connection: aio_pika.abc.AbstractConnection, payload: dict[str, Any]) -> None:
    channel = await connection.channel()
    exchange = await channel.declare_exchange(EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
    body = json.dumps(payload).encode("utf-8")
    await exchange.publish(
        aio_pika.Message(body=body, content_type="application/json"),
        routing_key=CANDIDATE_RK,
    )
    await channel.close()


async def run_smoke(amqp_url: str, *, pause_sec: float) -> int:
    print("=== AE Brain symbol-safety smoke test ===")
    print(f"AMQP: {amqp_url.split('@')[-1] if '@' in amqp_url else amqp_url}")
    print(f"Allowed universe: {','.join(sorted(ALLOWED))}")
    print()

    connection = await aio_pika.connect_robust(amqp_url)
    try:
        before_candidates = await _queue_depth(connection, CANDIDATE_QUEUE)
        before_signals = await _queue_depth(connection, SIGNAL_QUEUE)
        print(f"Queue depth before: {CANDIDATE_QUEUE}={before_candidates} {SIGNAL_QUEUE}={before_signals}")
        print()

        for label, payload, expected in SCENARIOS:
            symbol = payload["symbol"]
            if symbol not in ALLOWED:
                print(f"[REJECT upstream expected] {label}: symbol={symbol} -> {expected}")
            else:
                print(f"[PUBLISH] {label}: symbol={symbol}")
            try:
                await _publish(connection, payload)
                print(f"  published to {CANDIDATE_RK}")
            except Exception as exc:
                print(f"  publish failed: {exc}")
            print(f"  expected: {expected}")
            print()

        if pause_sec > 0:
            print(f"Waiting {pause_sec}s for ae-brain to consume...")
            await asyncio.sleep(pause_sec)

        after_candidates = await _queue_depth(connection, CANDIDATE_QUEUE)
        after_signals = await _queue_depth(connection, SIGNAL_QUEUE)
        print(f"Queue depth after:  {CANDIDATE_QUEUE}={after_candidates} {SIGNAL_QUEUE}={after_signals}")
        print()
        print("Verify in logs:")
        print("  binance-candidate-service: candidate_rejected_symbol (if publishing ADA)")
        print("  ae-brain: candidate_rejected_symbol for ADAUSDT")
        print("  ae-brain: confidence_below_threshold for low-confidence signals")
        print("  notification-service: telegram_signal_rejected for any gate violation")
        print()
        print("Only BTCUSDT/ETHUSDT with decision LONG/SHORT and confidence >= 0.70")
        print("should reach Telegram via signal.final.")
    finally:
        await connection.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test symbol safety pipeline")
    parser.add_argument(
        "--amqp-url",
        default=os.environ.get(
            "AEB_INPUT_AMQP_URL",
            os.environ.get("RABBITMQ_URL", "amqp://analeyes:analeyes_dev_secret@localhost:5672/analeyes"),
        ),
    )
    parser.add_argument("--pause-sec", type=float, default=5.0, help="Wait after publish for consumers")
    args = parser.parse_args()
    return asyncio.run(run_smoke(args.amqp_url, pause_sec=args.pause_sec))


if __name__ == "__main__":
    sys.exit(main())
