from __future__ import annotations
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any
import pika

def _utc_now_ms() -> int:
    return int(time.time() * 1000)

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _fmt_dt(value: Any) -> str:
    if not value:
        return '-'
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        dt = dt.astimezone(timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return str(value)

def _print_live(payload: dict[str, Any]) -> None:
    symbol = payload.get('symbol', '?')
    asset = payload.get('asset_class', '?')
    price = _safe_float(payload.get('price'))
    bid = _safe_float(payload.get('bid'))
    ask = _safe_float(payload.get('ask'))
    ts = payload.get('ts')
    lag_ms = None
    try:
        lag_ms = _utc_now_ms() - int(ts)
    except Exception:
        pass
    print(f'[LIVE] {_fmt_dt(payload.get('timestamp'))} | {symbol:<10} | {asset:<6} | price={(price if price is not None else '-'):>10} | bid={(bid if bid is not None else '-'):>10} | ask={(ask if ask is not None else '-'):>10} | lag_ms={(lag_ms if lag_ms is not None else '-')}', flush=True)

def _print_candidate(payload: dict[str, Any]) -> None:
    symbol = payload.get('symbol', '?')
    asset = payload.get('asset_class', '?')
    trigger = payload.get('trigger_reason', '-')
    trigger_reasons = payload.get('trigger_reasons')
    if isinstance(trigger_reasons, list) and trigger_reasons:
        trigger = ','.join((str(item) for item in trigger_reasons))
    consensus = payload.get('heuristic_signal_consensus', '-')
    score = payload.get('composite_score', '-')
    print(f'[AI  ] {_fmt_dt(payload.get('timestamp'))} | {symbol:<10} | {asset:<6} | trigger={trigger} | consensus={consensus} | score={score}', flush=True)

def main() -> int:
    parser = argparse.ArgumentParser(description='Human-friendly live output for external-markets-service streams.')
    parser.add_argument('--stream', choices=['all', 'live', 'ai'], default='all', help='What stream to show (default: all).')
    parser.add_argument('--rabbitmq-url', default=os.getenv('RABBITMQ_URL', 'amqp://user:password@rabbitmq:5672/'), help='AMQP URL (default from RABBITMQ_URL env).')
    parser.add_argument('--exchange', default=os.getenv('RABBITMQ_EXCHANGE', 'analeyes_exchange'), help='Exchange name (default: analeyes_exchange).')
    args = parser.parse_args()
    routing_keys: list[str]
    if args.stream == 'live':
        routing_keys = ['data.live_prices.external']
    elif args.stream == 'ai':
        routing_keys = ['data.candidates.ai']
    else:
        routing_keys = ['data.live_prices.external', 'data.candidates.ai']
    params = pika.URLParameters(args.rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange=args.exchange, exchange_type='topic', durable=True)
    queue = channel.queue_declare(queue='', exclusive=True, auto_delete=True)
    queue_name = queue.method.queue
    for key in routing_keys:
        channel.queue_bind(exchange=args.exchange, queue=queue_name, routing_key=key)
    print(f'Connected to {args.exchange}. Watching: {', '.join(routing_keys)}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    stopped = False

    def _stop(*_: Any) -> None:
        nonlocal stopped
        stopped = True
        try:
            channel.stop_consuming()
        except Exception:
            pass
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def _on_message(ch: pika.adapters.blocking_connection.BlockingChannel, method: pika.spec.Basic.Deliver, props: pika.spec.BasicProperties, body: bytes) -> None:
        _ = (props,)
        try:
            payload = json.loads(body.decode('utf-8'))
        except Exception as exc:
            print(f'[WARN] bad json: {exc}', flush=True)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        if method.routing_key == 'data.live_prices.external':
            _print_live(payload)
        elif method.routing_key == 'data.candidates.ai':
            _print_candidate(payload)
        else:
            print(f'[INFO] {method.routing_key}: {payload}', flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    channel.basic_qos(prefetch_count=100)
    channel.basic_consume(queue=queue_name, on_message_callback=_on_message, auto_ack=False)
    try:
        channel.start_consuming()
    finally:
        if not stopped:
            print('Stopping stream viewer.', flush=True)
        try:
            connection.close()
        except Exception:
            pass
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
