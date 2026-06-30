from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, Awaitable, Callable

import pika

from shared.config import Config
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from shared.rabbitmq_config import rabbitmq_connection_info, resolve_rabbitmq_url
from shared.utils.pika_client import PikaClient
from shared.utils.rabbitmq_topology import EXCHANGE, Queue, RoutingKey, declare_exchange

from src.logic.telegram.telegram_sender import TelegramSender

logging.basicConfig(level=logging.INFO, format='%(asctime)s [notification] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]

# queue_declare defaults used for both passive probe and active create.
_QUEUE_DURABLE = True
_QUEUE_EXCLUSIVE = False
_QUEUE_AUTO_DELETE = False


def _channel_close_details(channel: pika.adapters.blocking_connection.BlockingChannel) -> str:
    """Best-effort dump of why a pika channel was closed."""
    parts = [
        f'is_open={channel.is_open}',
        f'reply_code={getattr(channel, "reply_code", None)}',
        f'reply_text={getattr(channel, "reply_text", None)}',
    ]
    close_reason = getattr(channel, 'close_reason', None)
    if close_reason is not None:
        parts.append(f'close_reason={close_reason!r}')
    return ', '.join(parts)


class RabbitConsumer:
    """Blocking RabbitMQ consumer that runs in a dedicated thread with auto-reconnect."""

    def __init__(
        self,
        *,
        amqp_url: str,
        exchange: str,
        bindings: list[tuple[str, str, str]],
        make_callback: Callable[[str, str, str], Callable],
        stop_event: threading.Event,
        ready_event: threading.Event,
    ) -> None:
        self.amqp_url = amqp_url
        self.exchange = exchange
        # (queue_name, routing_key, consumer_tag_suffix)
        self.bindings = bindings
        self.make_callback = make_callback
        self.stop_event = stop_event
        self.ready_event = ready_event

    def run_forever(self) -> None:
        backoff_sec = 1
        while not self.stop_event.is_set():
            self.ready_event.clear()
            try:
                logger.info('RabbitMQ consumer session starting')
                self._run_session()
                logger.warning('RabbitMQ consumer session ended unexpectedly')
            except Exception:
                logger.exception('RabbitMQ consumer session crashed')
            finally:
                self.ready_event.clear()

            if self.stop_event.is_set():
                break

            logger.info('RabbitMQ consumer reconnecting in %ss', backoff_sec)
            time.sleep(backoff_sec)
            backoff_sec = min(backoff_sec * 2, 30)

        logger.info('RabbitMQ consumer thread exiting')

    def _open_channel(self, connection: pika.BlockingConnection) -> pika.adapters.blocking_connection.BlockingChannel:
        channel = connection.channel()
        if not channel.is_open:
            raise RuntimeError('RabbitMQ channel is not open after creation')
        declare_exchange(channel, self.exchange)
        if not channel.is_open:
            raise RuntimeError(
                f'Channel closed immediately after exchange_declare ({_channel_close_details(channel)})'
            )
        return channel

    def _ensure_queue(
        self,
        connection: pika.BlockingConnection,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        queue_name: str,
        routing_key: str,
    ) -> pika.adapters.blocking_connection.BlockingChannel:
        """Passive-check an existing queue first; declare only if missing."""
        logger.info(
            'queue_declare params: name=%s durable=%s exclusive=%s auto_delete=%s passive=True',
            queue_name,
            _QUEUE_DURABLE,
            _QUEUE_EXCLUSIVE,
            _QUEUE_AUTO_DELETE,
        )
        try:
            result = channel.queue_declare(queue=queue_name, passive=True)
            logger.info(
                'Passive queue_declare OK: name=%s message_count=%s consumer_count=%s channel_open=%s',
                queue_name,
                result.method.message_count,
                result.method.consumer_count,
                channel.is_open,
            )
        except pika.exceptions.AMQPChannelError as exc:
            logger.error(
                'Passive queue_declare FAILED: queue=%s error=%s %s',
                queue_name,
                exc,
                _channel_close_details(channel),
            )
            # Broker closes the channel on passive miss / arg mismatch; open a fresh one.
            channel = self._open_channel(connection)

            logger.info(
                'queue_declare params: name=%s durable=%s exclusive=%s auto_delete=%s passive=False',
                queue_name,
                _QUEUE_DURABLE,
                _QUEUE_EXCLUSIVE,
                _QUEUE_AUTO_DELETE,
            )
            try:
                channel.queue_declare(
                    queue=queue_name,
                    durable=_QUEUE_DURABLE,
                    exclusive=_QUEUE_EXCLUSIVE,
                    auto_delete=_QUEUE_AUTO_DELETE,
                )
            except pika.exceptions.AMQPChannelError as declare_exc:
                logger.error(
                    'Active queue_declare FAILED: queue=%s error=%s %s',
                    queue_name,
                    declare_exc,
                    _channel_close_details(channel),
                )
                raise

        if not channel.is_open:
            raise Exception(
                f'Channel closed immediately after queue_declare on {queue_name} '
                f'({_channel_close_details(channel)})'
            )

        channel.queue_bind(exchange=self.exchange, queue=queue_name, routing_key=routing_key)
        logger.info(
            'queue_bind OK: exchange=%s queue=%s routing_key=%s channel_open=%s',
            self.exchange,
            queue_name,
            routing_key,
            channel.is_open,
        )
        if not channel.is_open:
            raise Exception(
                f'Channel closed immediately after queue_bind on {queue_name} '
                f'({_channel_close_details(channel)})'
            )
        return channel

    def _register_consumer(
        self,
        connection: pika.BlockingConnection,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        queue_name: str,
        tag_suffix: str,
        pid: int,
    ) -> pika.adapters.blocking_connection.BlockingChannel:
        if not channel.is_open:
            raise RuntimeError(
                f'RabbitMQ channel is not open before basic_consume on {queue_name} '
                f'({_channel_close_details(channel)})'
            )

        consumer_tag = f'notification-{tag_suffix}-{pid}'
        callback = self.make_callback(queue_name, consumer_tag, tag_suffix)
        logger.info(
            'basic_consume params: queue=%s consumer_tag=%s auto_ack=False channel_open=%s',
            queue_name,
            consumer_tag,
            channel.is_open,
        )
        try:
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=False,
                consumer_tag=consumer_tag,
            )
        except pika.exceptions.AMQPChannelError as exc:
            logger.error(
                'basic_consume FAILED: queue=%s consumer_tag=%s error=%s %s',
                queue_name,
                consumer_tag,
                exc,
                _channel_close_details(channel),
            )
            raise

        if not channel.is_open:
            raise Exception(
                f'Channel closed immediately after registration on {queue_name} '
                f'({_channel_close_details(channel)})'
            )

        logger.info(
            'Successfully registered basic_consume queue=%s consumer_tag=%s channel_open=%s',
            queue_name,
            consumer_tag,
            channel.is_open,
        )
        return channel

    def _run_session(self) -> None:
        # Log vhost/host/user (password redacted) so we can spot wrong-broker connections.
        info = rabbitmq_connection_info(self.amqp_url)
        logger.info(
            'RabbitMQ connecting url=%s host=%s port=%s vhost=%s user=%s exchange=%s',
            info['url_sanitized'],
            info['host'],
            info['port'],
            info['vhost'],
            info['user'],
            self.exchange,
        )

        params = pika.URLParameters(self.amqp_url)
        params.heartbeat = 600
        params.blocked_connection_timeout = 300

        connection = pika.BlockingConnection(params)
        if not connection.is_open:
            raise RuntimeError('RabbitMQ connection is not open after connect')

        channel = self._open_channel(connection)

        for queue_name, routing_key, _tag_suffix in self.bindings:
            channel = self._ensure_queue(connection, channel, queue_name, routing_key)

        if not channel.is_open:
            raise RuntimeError(
                f'RabbitMQ channel closed after queue setup ({_channel_close_details(channel)})'
            )

        # Broker-side topology snapshot BEFORE registering consumers (passive declare
        # on a channel that already has active consumers can invalidate them).
        for queue_name, routing_key, _ in self.bindings:
            try:
                probe = channel.queue_declare(queue=queue_name, passive=True)
                logger.info(
                    'Topology snapshot queue=%s rk=%s exchange=%s messages=%s consumers=%s',
                    queue_name,
                    routing_key,
                    self.exchange,
                    probe.method.message_count,
                    probe.method.consumer_count,
                )
            except pika.exceptions.AMQPChannelError as exc:
                logger.error(
                    'Topology snapshot FAILED queue=%s error=%s %s',
                    queue_name,
                    exc,
                    _channel_close_details(channel),
                )
                raise

        channel.basic_qos(prefetch_count=1)

        pid = os.getpid()
        for queue_name, _routing_key, tag_suffix in self.bindings:
            channel = self._register_consumer(connection, channel, queue_name, tag_suffix, pid)

        if not channel.is_open:
            raise Exception(
                f'Channel closed immediately after all consumer registrations '
                f'({_channel_close_details(channel)})'
            )

        self.ready_event.set()
        logger.info(
            'Notification service consuming queues: %s, %s, %s (connection_open=%s channel_open=%s)',
            Queue.NEW_SIGNALS,
            Queue.SIGNAL_OUTCOMES,
            Queue.SIGNAL_ENTRY_EVENTS,
            connection.is_open,
            channel.is_open,
        )

        def _request_stop() -> None:
            if channel.is_open:
                channel.stop_consuming()

        def _watch_stop() -> None:
            while not self.stop_event.is_set():
                time.sleep(0.5)
            logger.info('Stop requested — cancelling channel.start_consuming()')
            with contextlib.suppress(Exception):
                connection.add_callback_threadsafe(_request_stop)

        threading.Thread(target=_watch_stop, name='notification-rabbit-stop', daemon=True).start()

        try:
            logger.info('Entering channel.start_consuming() — blocking until messages arrive')
            channel.start_consuming()
        except Exception:
            if not self.stop_event.is_set():
                logger.exception('channel.start_consuming() exited with error')
            raise
        finally:
            self.ready_event.clear()
            with contextlib.suppress(Exception):
                if channel.is_open:
                    channel.close()
            with contextlib.suppress(Exception):
                if connection.is_open:
                    connection.close()
            logger.info('RabbitMQ consumer session closed')


class NotificationServiceApp:
    EXCHANGE = EXCHANGE

    def __init__(self) -> None:
        self.config = Config('/app/config/settings.yml').all()
        self.sender = TelegramSender(self.config)
        self.amqp_url = resolve_rabbitmq_url()
        self._publisher = PikaClient(self.amqp_url, default_exchange=self.EXCHANGE)
        info = rabbitmq_connection_info(self.amqp_url)
        logger.info(
            'RabbitMQ startup url=%s host=%s port=%s vhost=%s user=%s exchange=%s',
            info['url_sanitized'],
            info['host'],
            info['port'],
            info['vhost'],
            info['user'],
            self.EXCHANGE,
        )
        for queue_name, routing_key in (
            (Queue.NEW_SIGNALS, RoutingKey.SIGNAL_FINAL),
            (Queue.SIGNAL_OUTCOMES, RoutingKey.SIGNAL_OUTCOME),
            (Queue.SIGNAL_ENTRY_EVENTS, RoutingKey.SIGNAL_ENTRY_EVENT),
        ):
            logger.info('RabbitMQ binding queue=%s routing_key=%s', queue_name, routing_key)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._consumer_ready = threading.Event()
        self._consumer_thread: threading.Thread | None = None
        self._consumer_crash: BaseException | None = None

    async def start(self) -> None:
        await DatabaseManager.initialize(database_url=os.environ.get('DATABASE_URL'))
        await self._publisher.connect()
        self._loop = asyncio.get_running_loop()
        self._start_consumer_thread()
        await self._wait_for_consumer_ready()
        logger.info('Notification service started; RabbitMQ consumers are registered')
        status_task = asyncio.create_task(self._status_heartbeat_loop())

        while not self._stop_event.is_set():
            if not self._consumer_thread or not self._consumer_thread.is_alive():
                crash = self._consumer_crash
                logger.critical(
                    'RabbitMQ consumer thread is not alive (crash=%s); exiting so orchestrator can restart',
                    crash,
                )
                status_task.cancel()
                raise SystemExit(1)
            if not self._consumer_ready.is_set():
                logger.warning('RabbitMQ consumer lost readiness; waiting for reconnect')
            await asyncio.sleep(2)

    async def _status_heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = json.dumps(
                    {'service': 'notification-service', 'status': 'ok', 'ts': time.time()},
                    ensure_ascii=False,
                )
                await self._publisher.publish_async(self.EXCHANGE, RoutingKey.STATUS_NOTIFICATION, payload)
            except Exception:
                logger.exception('Failed to publish status.notification heartbeat')
            await asyncio.sleep(5)

    def _start_consumer_thread(self) -> None:
        self._consumer_crash = None

        def _thread_main() -> None:
            try:
                consumer = RabbitConsumer(
                    amqp_url=self.amqp_url,
                    exchange=self.EXCHANGE,
                    bindings=[
                        (Queue.NEW_SIGNALS, RoutingKey.SIGNAL_FINAL, 'signal-final'),
                        (Queue.SIGNAL_OUTCOMES, RoutingKey.SIGNAL_OUTCOME, 'outcome'),
                        (Queue.SIGNAL_ENTRY_EVENTS, RoutingKey.SIGNAL_ENTRY_EVENT, 'entry'),
                    ],
                    make_callback=self._make_message_callback,
                    stop_event=self._stop_event,
                    ready_event=self._consumer_ready,
                )
                consumer.run_forever()
            except Exception as exc:
                self._consumer_crash = exc
                logger.exception('RabbitMQ consumer thread terminated with error')
                raise

        self._consumer_thread = threading.Thread(
            target=_thread_main,
            name='notification-rabbit-consumer',
            daemon=False,
        )
        self._consumer_thread.start()
        logger.info('RabbitMQ consumer thread launched')

    async def _wait_for_consumer_ready(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self._consumer_ready.is_set():
                return
            if self._consumer_thread and not self._consumer_thread.is_alive():
                raise RuntimeError(
                    f'RabbitMQ consumer thread died during startup (crash={self._consumer_crash})'
                )
            await asyncio.sleep(0.25)
        raise TimeoutError('RabbitMQ consumer failed to register within 30 seconds')

    def _make_message_callback(
        self,
        queue_name: str,
        consumer_tag: str,
        tag_suffix: str,
    ) -> Callable:
        handler = {
            'signal-final': self._handle_signal,
            'outcome': self._handle_outcome,
            'entry': self._handle_entry,
        }[tag_suffix]

        def _on_message(
            ch: pika.adapters.blocking_connection.BlockingChannel,
            method: pika.spec.Basic.Deliver,
            properties: pika.BasicProperties,
            body: bytes,
        ) -> None:
            headers = dict(properties.headers) if properties and properties.headers else {}
            if queue_name == Queue.NEW_SIGNALS:
                logger.info(
                    'Received message from %s delivery_tag=%s exchange=%r routing_key=%r',
                    Queue.NEW_SIGNALS,
                    method.delivery_tag,
                    method.exchange,
                    method.routing_key,
                )
            logger.info('[DEBUG] Received message body: %s', body)
            logger.info(
                '[DEBUG] AMQP frame: consumer_tag=%s queue=%s delivery_tag=%s '
                'exchange=%r routing_key=%r redelivered=%s headers=%s content_type=%s',
                getattr(method, 'consumer_tag', consumer_tag),
                queue_name,
                method.delivery_tag,
                method.exchange,
                method.routing_key,
                getattr(method, 'redelivered', None),
                headers,
                getattr(properties, 'content_type', None) if properties else None,
            )
            try:
                payload = json.loads(body.decode('utf-8'))
                if self._loop is None:
                    raise RuntimeError('Main asyncio loop is not initialized')

                future = asyncio.run_coroutine_threadsafe(
                    self._dispatch(handler, payload),
                    self._loop,
                )
                future.result(timeout=120)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    '[DEBUG] Acked message consumer=%s delivery_tag=%s',
                    consumer_tag,
                    method.delivery_tag,
                )
            except json.JSONDecodeError:
                logger.exception('[DEBUG] Invalid JSON on queue=%s', queue_name)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception:
                logger.exception('[DEBUG] Handler failed on queue=%s', queue_name)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        return _on_message

    async def _dispatch(self, handler: Handler, payload: dict[str, Any]) -> None:
        await handler(payload)

    async def _handle_signal(self, payload: dict[str, Any]) -> None:
        normalized = TelegramSender.normalize_signal_payload(payload)
        symbol = normalized.get("symbol")
        source_ai = normalized.get("source_ai")
        decision = normalized.get("decision")

        logger.info(
            "Resolved signal.final symbol=%s decision=%s source_ai=%s",
            symbol,
            decision,
            source_ai,
        )
        logger.info(
            "Telegram config token_present=%s group_id_present=%s",
            bool(self.sender.bot_token),
            bool(self.sender.group_id),
        )
        topic_id = self.sender.resolve_topic_id(normalized.get("asset_class"), source_ai)
        logger.info(
            "Telegram topic mapping source_ai=%s topic_id=%s allow_no_topic=%s",
            source_ai,
            topic_id,
            self.sender.allow_no_topic,
        )

        logger.info("Sending Telegram signal...")
        result = await self.sender.send_signal(normalized)
        logger.info(
            "Telegram send result sent=%s skip_reason=%s",
            result.sent,
            result.skip_reason,
        )
        await self._update_signal_log(
            normalized,
            telegram_sent=result.sent,
            skip_reason=result.skip_reason,
        )

    async def _handle_outcome(self, payload: dict[str, Any]) -> None:
        await self.sender.send_signal_outcome(payload)

    async def _handle_entry(self, payload: dict[str, Any]) -> None:
        await self.sender.send_entry_event_notification(payload)

    async def _update_signal_log(
        self,
        payload: dict[str, Any],
        *,
        telegram_sent: bool,
        skip_reason: str | None = None,
    ) -> None:
        db_id = payload.get('signal_log_db_id')
        if not db_id:
            return
        try:
            async for session in DatabaseManager.get_session():
                row = await session.get(SignalFeatureLog, int(db_id))
                if row is None:
                    return
                row.telegram_message_sent = telegram_sent
                row.ai_signal_type = payload.get('decision')
                row.ai_confidence = payload.get('confidence')
                summary = payload.get('reason')
                if skip_reason and not telegram_sent:
                    summary = f"{summary or ''} [telegram_skipped: {skip_reason}]".strip()
                row.ai_reason_summary = summary
                row.ai_entry_price_suggestion = (
                    payload.get('entry_price') if isinstance(payload.get('entry_price'), (int, float)) else None
                )
                row.ai_tp_price_suggestion = payload.get('tp')
                row.ai_sl_price_suggestion = payload.get('sl')
                row.ai_leverage_suggestion = payload.get('leverage')
                row.ai_consensus_achieved = payload.get('consensus_achieved')
        except Exception:
            logger.exception('Failed to update signal_feature_logs for id=%s', db_id)


async def main() -> None:
    app = NotificationServiceApp()
    await app.start()


if __name__ == '__main__':
    asyncio.run(main())
