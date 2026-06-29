from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pika
from pika.adapters.asyncio_connection import AsyncioConnection

from shared.config import Config
from shared.database.db_manager import DatabaseManager
from shared.database.models import EnsembleModelDecision
from shared.utils.data_encoder import dumps_payload

from src.logic.analyzer import AssetClassAwareAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [ai-service] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class AIServiceApp:
    """AI-сервис: потребляет data.candidates.ai, публикует signal.final."""

    EXCHANGE = 'analeyes_exchange'
    IN_QUEUE = 'q_ai_candidates'
    IN_RK = 'data.candidates.ai'
    OUT_RK = 'signal.final'

    def __init__(self) -> None:
        self.config = Config('/app/config/settings.yml')
        self.amqp_url = os.environ.get('RABBITMQ_URL', 'amqp://user:password@rabbitmq:5672/')
        prompts_dir = Path('/app/config/prompts')
        min_composite = float(self.config.get('ai_engines.composite_threshold', 0.5))
        self.analyzer = AssetClassAwareAnalyzer(prompts_dir=prompts_dir, min_composite=min_composite)
        self._connection: AsyncioConnection | None = None
        self._channel: pika.channel.Channel | None = None

    async def start(self) -> None:
        await DatabaseManager.initialize(database_url=os.environ.get('DATABASE_URL'))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._connect_blocking)
        logger.info('AI service started')

    def _connect_blocking(self) -> None:
        params = pika.URLParameters(self.amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.exchange_declare(exchange=self.EXCHANGE, exchange_type='topic', durable=True)
        channel.queue_declare(queue=self.IN_QUEUE, durable=True)
        channel.queue_bind(exchange=self.EXCHANGE, queue=self.IN_QUEUE, routing_key=self.IN_RK)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.IN_QUEUE, on_message_callback=self._on_message, auto_ack=False)
        logger.info('Consuming %s', self.IN_RK)
        channel.start_consuming()

    def _on_message(self, channel: pika.channel.Channel, method: pika.spec.Basic.Deliver, properties: pika.BasicProperties, body: bytes) -> None:
        try:
            candidate = json.loads(body.decode('utf-8'))
            result = self.analyzer.analyze(candidate)
            if result.get('decision') == 'SKIP':
                logger.info('SKIP candidate symbol=%s asset_class=%s', candidate.get('symbol'), candidate.get('asset_class'))
                return
            final_payload = self._build_final_payload(candidate, result)
            channel.basic_publish(
                exchange=self.EXCHANGE,
                routing_key=self.OUT_RK,
                body=dumps_payload(final_payload),
                properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'),
            )
            asyncio.run(self._persist_decision(candidate, result))
            logger.info(
                'Published signal.final symbol=%s decision=%s asset_class=%s',
                final_payload.get('symbol'),
                final_payload.get('decision'),
                final_payload.get('asset_class'),
            )
        except Exception as exc:
            logger.error('Candidate handler error: %s', exc, exc_info=True)
        finally:
            try:
                channel.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as ack_exc:
                logger.error('Ack failed: %s', ack_exc)

    def _build_final_payload(self, candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        signal_id = candidate.get('signal_id') or str(uuid.uuid4())
        return {
            'symbol': candidate.get('symbol'),
            'name': candidate.get('name'),
            'asset_class': candidate.get('asset_class', 'crypto'),
            'signal_id': signal_id,
            'source_ai': 'ensemble',
            'decision': result['decision'],
            'confidence': result['confidence'],
            'reason': result.get('reason'),
            'entry_price': result.get('entry_price', 'market'),
            'tp': result.get('tp'),
            'sl': result.get('sl'),
            'leverage': result.get('leverage', 1.0),
            'composite_score_data': {'composite_score': candidate.get('composite_score')},
            'signal_log_db_id': candidate.get('signal_log_db_id'),
            'historical_ohlcv': candidate.get('historical_ohlcv'),
            'features': candidate.get('features'),
        }

    async def _persist_decision(self, candidate: dict[str, Any], result: dict[str, Any]) -> None:
        try:
            async for session in DatabaseManager.get_session():
                row = EnsembleModelDecision(
                    asset_class=str(candidate.get('asset_class', 'crypto')),
                    signal_id=uuid.UUID(str(candidate.get('signal_id'))),
                    symbol=str(candidate.get('symbol')),
                    model='ensemble',
                    timestamp=__import__('datetime').datetime.now(tz=__import__('datetime').timezone.utc),
                    decision=str(result.get('decision')),
                    confidence=float(result.get('confidence') or 0.0),
                    raw_response=json.dumps(result, ensure_ascii=False),
                )
                session.add(row)
        except Exception as exc:
            logger.error('Failed to persist ensemble_model_decisions: %s', exc)


async def main() -> None:
    app = AIServiceApp()
    await app.start()


if __name__ == '__main__':
    asyncio.run(main())
