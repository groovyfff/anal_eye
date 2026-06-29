from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pika
import yaml
from dotenv import load_dotenv
from shared.database.db_manager import DatabaseManager
from shared.market_hours import MarketHours
from shared.utils.data_encoder import dumps_payload

from src.logic.external_prices import ExternalPriceStore
from src.logic.signal_tracker import SignalTracker, TrackedSignal
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class _MetricStub:
    def inc(self, amount: float = 1.0) -> None:
        return None

    def set(self, value: float) -> None:
        return None

    def observe(self, value: float) -> None:
        return None


def _build_metrics() -> dict[str, Any]:
    try:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server

        return {
            'enabled': True,
            'start_http_server': start_http_server,
            'tracker_active_signals': Gauge('tracker_active_signals', 'Active tracked signals'),
            'tracker_duplicate_signal_total': Counter('tracker_duplicate_signal_total', 'Duplicate signal.final messages'),
            'tracker_tp_hit_total': Counter('tracker_tp_hit_total', 'TP outcomes'),
            'tracker_sl_hit_total': Counter('tracker_sl_hit_total', 'SL outcomes'),
            'tracker_expired_signal_total': Counter('tracker_expired_signal_total', 'Expired outcomes'),
            'tracker_cancelled_unfilled_total': Counter('tracker_cancelled_unfilled_total', 'Cancelled unfilled outcomes'),
            'ws_price_age_ms': Histogram('ws_price_age_ms', 'Price freshness age in ms'),
        }
    except ImportError:
        logger.warning('[tracker] prometheus_client недоступен — метрики отключены')
        stub = _MetricStub()
        return {
            'enabled': False,
            'start_http_server': lambda _port: None,
            'tracker_active_signals': stub,
            'tracker_duplicate_signal_total': stub,
            'tracker_tp_hit_total': stub,
            'tracker_sl_hit_total': stub,
            'tracker_expired_signal_total': stub,
            'tracker_cancelled_unfilled_total': stub,
            'ws_price_age_ms': stub,
        }


class RabbitPublisher:
    """Минимальный async-friendly publisher (аналог external-markets PikaClient)."""

    def __init__(self, url: str, default_exchange: str) -> None:
        self.url = url
        self.default_exchange = default_exchange
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tracker-pika')
        self._thread_id: int | None = None

    async def connect(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._connect_blocking)
            return True
        except Exception as exc:
            logger.error('[tracker] RabbitMQ connect failed: %s', exc)
            return False

    def _connect_blocking(self) -> None:
        self._thread_id = threading.get_ident()
        params = pika.URLParameters(self.url)
        self._connection = pika.BlockingConnection(parameters=params)
        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=self.default_exchange, exchange_type='topic', durable=True)

    def _ensure_connected(self) -> None:
        if self._connection and self._connection.is_open and self._channel is not None:
            return
        self._connect_blocking()

    def _publish_blocking(self, exchange_name: str, routing_key: str, body: str) -> None:
        self._ensure_connected()
        if self._channel is None:
            raise RuntimeError('RabbitMQ channel is not available')
        self._channel.basic_publish(
            exchange=exchange_name,
            routing_key=routing_key,
            body=body.encode('utf-8'),
            properties=pika.BasicProperties(content_type='application/json', delivery_mode=2),
        )

    async def publish_async(self, exchange_name: str, routing_key: str, body: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._publish_blocking, exchange_name, routing_key, body)

    async def close(self) -> None:
        if self._connection and self._connection.is_open:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._connection.close)
        self._executor.shutdown(wait=False, cancel_futures=False)


class RabbitConsumer:
    """Blocking consumer в отдельном потоке; колбэки пробрасываются в asyncio loop."""

    def __init__(
        self,
        url: str,
        exchange: str,
        bindings: list[tuple[str, str]],
        on_message: Callable[[str, dict[str, Any]], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.url = url
        self.exchange = exchange
        self.bindings = bindings
        self.on_message = on_message
        self.loop = loop
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name='tracker-rabbit-consumer', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        params = pika.URLParameters(self.url)
        connection = pika.BlockingConnection(parameters=params)
        channel = connection.channel()
        channel.exchange_declare(exchange=self.exchange, exchange_type='topic', durable=True)

        def _callback(ch: Any, method: Any, _properties: Any, body: bytes) -> None:
            try:
                payload = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                logger.error('[tracker] Невалидный JSON routing_key=%s', method.routing_key)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            routing_key = str(method.routing_key)
            self.loop.call_soon_threadsafe(self.on_message, routing_key, payload)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        for queue_name, routing_key in self.bindings:
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(exchange=self.exchange, queue=queue_name, routing_key=routing_key)
            channel.basic_consume(queue=queue_name, on_message_callback=_callback, auto_ack=False)

        while not self._stop_event.is_set() and connection.is_open:
            connection.process_data_events(time_limit=1)
        with contextlib.suppress(Exception):
            connection.close()


def load_settings(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    settings_path = Path(path)
    payload = yaml.safe_load(settings_path.read_text(encoding='utf-8')) or {}
    rabbitmq_url = os.getenv('RABBITMQ_URL')
    if rabbitmq_url:
        payload.setdefault('rabbitmq', {})['url'] = rabbitmq_url
    return payload


class SignalServiceApp:
    """Главное приложение tracker-service."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        setup_logging(settings.get('logging', {}))
        self.exchange_name = str(settings.get('rabbitmq', {}).get('exchange', 'analeyes_exchange'))
        tracker_cfg = settings.get('signal_tracker', {}) or {}
        self.check_interval_s = float(tracker_cfg.get('check_interval_s', 1))
        self.db_enabled = bool(settings.get('database', {}).get('enabled', True))
        self.market_hours = MarketHours(settings.get('market_hours', {}))
        max_age_ms = int(tracker_cfg.get('max_age_ms', 4500))
        self.price_store = ExternalPriceStore(max_age_ms=max_age_ms)
        self.metrics = _build_metrics()
        self.tracker = SignalTracker(
            market_hours=self.market_hours,
            price_store=self.price_store,
            entry_timeout_sec=int(tracker_cfg.get('entry_timeout_sec', 300)),
            expiration_hours=int(tracker_cfg.get('expiration_hours', 24)),
            slippage_pct=float(tracker_cfg.get('slippage_pct', 0.001)),
            max_age_ms=max_age_ms,
            prefer=str(tracker_cfg.get('prefer', 'tp')),
            default_bank_usd=float(settings.get('trading', {}).get('default_initial_bank_usd', 500)),
            db_enabled=self.db_enabled,
            on_entry=self._publish_entry_event,
            on_outcome=self._publish_outcome,
            on_duplicate=self._on_duplicate_signal,
        )
        rabbit_cfg = settings.get('rabbitmq', {}) or {}
        self.live_prices_routing_key = 'data.live_prices.external'
        self.signal_final_routing_key = 'signal.final'
        self.outcome_routing_key = 'signal.outcome'
        self.entry_event_routing_key = 'signal.entry_event'
        self.live_prices_queue = str(rabbit_cfg.get('live_prices_queue', self.live_prices_routing_key))
        self.signal_final_queue = str(rabbit_cfg.get('signal_final_queue', 'q_new_signals_for_tracker'))
        self.rabbit_url = rabbit_cfg.get('url', os.environ.get('RABBITMQ_URL', 'amqp://user:password@rabbitmq:5672/'))
        self.rabbit_connect_retries = int(rabbit_cfg.get('connect_retries', 10))
        self.rabbit_connect_retry_delay_s = float(rabbit_cfg.get('connect_retry_delay_s', 2.0))
        self.publisher = RabbitPublisher(url=self.rabbit_url, default_exchange=self.exchange_name)
        self._consumer: RabbitConsumer | None = None
        self._shutdown_event = asyncio.Event()
        self._pending_messages: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    def _on_duplicate_signal(self) -> None:
        self.metrics['tracker_duplicate_signal_total'].inc()

    async def _publish_entry_event(self, signal: TrackedSignal) -> None:
        payload = self.tracker.build_entry_event_payload(signal)
        await self.publisher.publish_async(
            exchange_name=self.exchange_name,
            routing_key=self.entry_event_routing_key,
            body=dumps_payload(payload),
        )
        logger.info('[tracker] Опубликован signal.entry_event signal_id=%s', signal.signal_id)

    async def _publish_outcome(
        self,
        signal: TrackedSignal,
        pnl_usdt: float | None,
        pnl_percent: float | None,
    ) -> None:
        payload = self.tracker.build_outcome_payload(signal, pnl_usdt, pnl_percent)
        await self.publisher.publish_async(
            exchange_name=self.exchange_name,
            routing_key=self.outcome_routing_key,
            body=dumps_payload(payload),
        )
        status = signal.state.value
        if status == 'TP_HIT':
            self.metrics['tracker_tp_hit_total'].inc()
        elif status == 'SL_HIT':
            self.metrics['tracker_sl_hit_total'].inc()
        elif status == 'EXPIRED':
            self.metrics['tracker_expired_signal_total'].inc()
        elif status == 'CANCELLED_UNFILLED':
            self.metrics['tracker_cancelled_unfilled_total'].inc()
        logger.info('[tracker] Опубликован signal.outcome signal_id=%s status=%s', signal.signal_id, status)

    def _enqueue_rabbit_message(self, routing_key: str, payload: dict[str, Any]) -> None:
        self._pending_messages.put_nowait((routing_key, payload))

    async def _dispatch_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                routing_key, payload = await asyncio.wait_for(self._pending_messages.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if routing_key == self.signal_final_routing_key:
                self.tracker.start_tracking_signal(payload)
                self.metrics['tracker_active_signals'].set(len(self.tracker.active_tracked_signals))
            elif routing_key == self.live_prices_routing_key:
                accepted = self.price_store.upsert_external_message(payload)
                if accepted:
                    ts_ms = int(payload.get('ts') or 0)
                    if ts_ms:
                        now_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
                        self.metrics['ws_price_age_ms'].observe(max(0, now_ms - ts_ms))
            else:
                logger.debug('[tracker] Неизвестный routing_key=%s', routing_key)

    async def _check_loop(self) -> None:
        while not self._shutdown_event.is_set():
            started = asyncio.get_running_loop().time()
            try:
                market_data_map = self.price_store.build_market_data_map()
                await self.tracker.check_tracked_signals(market_data_map)
                self.metrics['tracker_active_signals'].set(len(self.tracker.active_tracked_signals))
            except Exception as exc:
                logger.error('[tracker] Ошибка check_loop: %s', exc)
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, self.check_interval_s - elapsed))

    async def run(self) -> None:
        if self.db_enabled:
            db_cfg = self.settings.get('database', {}) or {}
            await DatabaseManager.initialize(
                database_url=os.environ.get('DATABASE_URL'),
                pool_size=int(db_cfg.get('pool_size', 5)),
                max_overflow=int(db_cfg.get('max_overflow', 10)),
            )
        prom_cfg = self.settings.get('prometheus', {}) or {}
        if prom_cfg.get('enabled', True):
            port = int(prom_cfg.get('port', 8000))
            self.metrics['start_http_server'](port)
            logger.info('[tracker] Prometheus /metrics на порту %s', port)
        connected = False
        for attempt in range(1, self.rabbit_connect_retries + 1):
            connected = await self.publisher.connect()
            if connected:
                break
            logger.warning(
                '[tracker] RabbitMQ недоступен, retry %s/%s',
                attempt,
                self.rabbit_connect_retries,
            )
            if attempt < self.rabbit_connect_retries:
                await asyncio.sleep(self.rabbit_connect_retry_delay_s)
        if not connected:
            raise RuntimeError('[tracker] Не удалось подключиться к RabbitMQ')
        loop = asyncio.get_running_loop()
        self._consumer = RabbitConsumer(
            url=self.rabbit_url,
            exchange=self.exchange_name,
            bindings=[
                (self.signal_final_queue, self.signal_final_routing_key),
                (self.live_prices_queue, self.live_prices_routing_key),
            ],
            on_message=self._enqueue_rabbit_message,
            loop=loop,
        )
        self._consumer.start()
        logger.info(
            '[tracker] Сервис запущен queues=%s,%s',
            self.signal_final_queue,
            self.live_prices_queue,
        )
        dispatch_task = asyncio.create_task(self._dispatch_loop(), name='dispatch_loop')
        check_task = asyncio.create_task(self._check_loop(), name='check_loop')
        await self._shutdown_event.wait()
        for task in (dispatch_task, check_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._consumer:
            self._consumer.stop()
        await self.publisher.close()
        if self.db_enabled:
            await DatabaseManager.close()

    async def stop(self) -> None:
        self._shutdown_event.set()


async def _run_service() -> None:
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'settings.yml'
    settings = load_settings(config_path)
    app = SignalServiceApp(settings)
    try:
        await app.run()
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        await app.stop()
    finally:
        await app.stop()


if __name__ == '__main__':
    try:
        asyncio.run(_run_service())
    except KeyboardInterrupt:
        pass
