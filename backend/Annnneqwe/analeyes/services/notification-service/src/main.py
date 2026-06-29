from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import pika
import yaml

from shared.config import Config
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from sqlalchemy import select

from src.logic.telegram_sender import TelegramSender

logging.basicConfig(level=logging.INFO, format='%(asctime)s [notification] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class NotificationServiceApp:
    EXCHANGE = 'analeyes_exchange'

    def __init__(self) -> None:
        self.config = Config('/app/config/settings.yml').all()
        self.sender = TelegramSender(self.config)
        self.amqp_url = os.environ.get('RABBITMQ_URL', 'amqp://user:password@rabbitmq:5672/')

    async def start(self) -> None:
        await DatabaseManager.initialize(database_url=os.environ.get('DATABASE_URL'))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._consume_blocking)

    def _consume_blocking(self) -> None:
        params = pika.URLParameters(self.amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.exchange_declare(exchange=self.EXCHANGE, exchange_type='topic', durable=True)
        for queue, rk, handler_name in [
            ('q_new_signals', 'signal.final', 'signal'),
            ('q_signal_outcomes', 'signal.outcome', 'outcome'),
            ('q_signal_entry_events', 'signal.entry_event', 'entry'),
        ]:
            channel.queue_declare(queue=queue, durable=True)
            channel.queue_bind(exchange=self.EXCHANGE, queue=queue, routing_key=rk)
        channel.basic_qos(prefetch_count=1)

        def _wrap(handler):
            def _on_message(ch, method, properties, body):
                try:
                    payload = json.loads(body.decode('utf-8'))
                    asyncio.run(handler(payload))
                    if handler.__name__ == 'handle_signal':
                        asyncio.run(self._update_signal_log(payload))
                except Exception as exc:
                    logger.error('Notification handler error: %s', exc, exc_info=True)
                finally:
                    try:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as ack_exc:
                        logger.error('Ack failed: %s', ack_exc)

            return _on_message

        channel.basic_consume('q_new_signals', on_message_callback=_wrap(self._handle_signal))
        channel.basic_consume('q_signal_outcomes', on_message_callback=_wrap(self._handle_outcome))
        channel.basic_consume('q_signal_entry_events', on_message_callback=_wrap(self._handle_entry))
        logger.info('Notification service consuming signal.final/outcome/entry_event')
        channel.start_consuming()

    async def _handle_signal(self, payload: dict[str, Any]) -> None:
        await self.sender.send_signal(payload)

    async def _handle_outcome(self, payload: dict[str, Any]) -> None:
        await self.sender.send_outcome(payload)

    async def _handle_entry(self, payload: dict[str, Any]) -> None:
        await self.sender.send_entry_event(payload)

    async def _update_signal_log(self, payload: dict[str, Any]) -> None:
        db_id = payload.get('signal_log_db_id')
        if not db_id:
            return
        try:
            async for session in DatabaseManager.get_session():
                row = await session.get(SignalFeatureLog, int(db_id))
                if row is None:
                    return
                row.telegram_message_sent = True
                row.ai_signal_type = payload.get('decision')
                row.ai_confidence = payload.get('confidence')
                row.ai_reason_summary = payload.get('reason')
                row.ai_entry_price_suggestion = payload.get('entry_price') if isinstance(payload.get('entry_price'), (int, float)) else None
                row.ai_tp_price_suggestion = payload.get('tp')
                row.ai_sl_price_suggestion = payload.get('sl')
                row.ai_leverage_suggestion = payload.get('leverage')
        except Exception as exc:
            logger.error('Failed to update signal_feature_logs: %s', exc)


async def main() -> None:
    app = NotificationServiceApp()
    await app.start()


if __name__ == '__main__':
    asyncio.run(main())
