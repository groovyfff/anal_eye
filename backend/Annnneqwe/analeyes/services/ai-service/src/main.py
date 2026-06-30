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
from shared.rabbitmq_config import rabbitmq_connection_info, resolve_rabbitmq_url
from shared.utils.data_encoder import dumps_payload
from shared.utils.rabbitmq_topology import EXCHANGE, Queue, RoutingKey, bind_queue, declare_exchange

from src.logic.analyzer import AssetClassAwareAnalyzer
from src.logic.candidate_normalizer import normalize_candidate, normalized_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s [ai-service] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _accept_manual_test_candidates() -> bool:
    return os.environ.get('AI_ACCEPT_MANUAL_TEST_CANDIDATES', '').strip().lower() in {
        '1',
        'true',
        'yes',
    }


def _consume_candidates_enabled() -> bool:
    return os.environ.get('AI_SERVICE_CONSUME_CANDIDATES', 'false').strip().lower() in {
        '1',
        'true',
        'yes',
    }


class AIServiceApp:
    """AI-сервис: потребляет data.candidates.ai, публикует signal.final."""

    EXCHANGE = EXCHANGE
    IN_QUEUE = Queue.DATA_CANDIDATES_AI
    IN_RK = RoutingKey.DATA_CANDIDATES_AI
    OUT_RK = RoutingKey.SIGNAL_FINAL
    BACKTEST_QUEUE = Queue.AI_BACKTEST_ANALYZE
    BACKTEST_RK = RoutingKey.AI_BACKTEST_ANALYZE

    def __init__(self) -> None:
        self.config = Config('/app/config/settings.yml')
        self.amqp_url = resolve_rabbitmq_url()
        self.accept_manual_test = _accept_manual_test_candidates()
        self.consume_candidates = _consume_candidates_enabled()
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
        logger.info('AI_ACCEPT_MANUAL_TEST_CANDIDATES=%s', self.accept_manual_test)
        logger.info('AI_SERVICE_CONSUME_CANDIDATES=%s', self.consume_candidates)
        if not self.consume_candidates:
            logger.info(
                'ai-service candidate consumer disabled; AE Brain is the active candidate processor'
            )
        prompts_dir = Path('/app/config/prompts')
        min_composite = float(self.config.get('ai_engines.composite_threshold', 0.5))
        self.analyzer = AssetClassAwareAnalyzer(prompts_dir=prompts_dir, min_composite=min_composite)
        self._connection: AsyncioConnection | None = None
        self._channel: pika.channel.Channel | None = None

    async def start(self) -> None:
        await DatabaseManager.initialize(database_url=os.environ.get('DATABASE_URL'))
        if not self.consume_candidates:
            logger.info(
                'ai-service idle: not consuming %s (AE Brain owns candidate processing)',
                self.IN_QUEUE,
            )
            while True:
                await asyncio.sleep(3600)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._connect_blocking)
        logger.info('AI service started')

    def _connect_blocking(self) -> None:
        params = pika.URLParameters(self.amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        declare_exchange(channel, self.EXCHANGE)
        channel.queue_declare(queue=self.IN_QUEUE, durable=True)
        bind_queue(channel, self.IN_QUEUE, self.IN_RK, self.EXCHANGE)
        channel.queue_declare(queue=self.BACKTEST_QUEUE, durable=True)
        bind_queue(channel, self.BACKTEST_QUEUE, self.BACKTEST_RK, self.EXCHANGE)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.IN_QUEUE, on_message_callback=self._on_message, auto_ack=False)
        logger.info('Consuming queue=%s routing_key=%s', self.IN_QUEUE, self.IN_RK)
        channel.start_consuming()

    def _on_message(
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            raw = json.loads(body.decode('utf-8'))
            logger.info(
                'Received data.candidates.ai payload_type=%s',
                type(raw).__name__,
            )
            candidate = normalize_candidate(raw)

            if candidate.get('_invalid_payload'):
                skip_reason = str(candidate.get('_invalid_reason') or 'invalid_payload')
                logger.info(
                    'SKIP candidate symbol=%s reason=%s normalized=%s',
                    candidate.get('symbol'),
                    skip_reason,
                    normalized_summary(candidate),
                )
                return

            logger.info(
                'Received data.candidates.ai symbol=%s asset_class=%s',
                candidate.get('symbol'),
                candidate.get('asset_class'),
            )
            logger.info(
                'Normalized candidate symbol=%s price=%s composite=%s candles_count=%s direction_hint=%s',
                candidate.get('symbol'),
                candidate.get('current_price'),
                candidate.get('composite_score'),
                len(candidate.get('candles') or []),
                candidate.get('direction_hint'),
            )

            result = self.analyzer.analyze(candidate, accept_manual_test=self.accept_manual_test)

            if result.get('decision') == 'SKIP':
                logger.info(
                    'Internal AI SKIP candidate symbol=%s reason=%s normalized=%s result=%s',
                    candidate.get('symbol'),
                    result.get('skip_reason') or result.get('reason'),
                    normalized_summary(candidate),
                    result,
                )
                return

            final_payload = self._build_final_payload(candidate, result)
            channel.basic_publish(
                exchange=self.EXCHANGE,
                routing_key=self.OUT_RK,
                body=dumps_payload(final_payload),
                properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'),
            )
            asyncio.run(self._persist_decision(final_payload, result))
            logger.info(
                'Published signal.final symbol=%s decision=%s source_ai=%s',
                final_payload.get('symbol'),
                final_payload.get('decision'),
                final_payload.get('source_ai'),
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
        decision = result['decision']
        source_ai = str(result.get('source_ai') or 'ensemble')
        reason_summary = result.get('reason_summary') or result.get('reason')
        return {
            'symbol': candidate.get('symbol'),
            'name': candidate.get('name'),
            'asset_class': candidate.get('asset_class', 'crypto'),
            'signal_id': signal_id,
            'source_ai': source_ai,
            'decision': decision,
            'signal_type': result.get('signal_type') or decision,
            'confidence': result['confidence'],
            'reason': reason_summary,
            'reason_summary': reason_summary,
            'entry_price': result.get('entry_price', 'market'),
            'tp': result.get('tp'),
            'tp_price': result.get('tp_price') or result.get('tp'),
            'sl': result.get('sl'),
            'sl_price': result.get('sl_price') or result.get('sl'),
            'leverage': result.get('leverage', 1.0),
            'composite_score_data': {'composite_score': candidate.get('composite_score')},
            'signal_log_db_id': candidate.get('signal_log_db_id'),
            'historical_ohlcv': candidate.get('historical_ohlcv') or candidate.get('candles'),
            'features': candidate.get('features'),
            'consensus_achieved': result.get('consensus_achieved'),
            'manual_test': result.get('manual_test', False),
        }

    async def _persist_decision(self, final_payload: dict[str, Any], result: dict[str, Any]) -> None:
        if self.accept_manual_test and final_payload.get('manual_test'):
            logger.info(
                'Skipping DB persist for manual test candidate symbol=%s signal_id=%s',
                final_payload.get('symbol'),
                final_payload.get('signal_id'),
            )
            return
        try:
            async for session in DatabaseManager.get_session():
                row = EnsembleModelDecision(
                    asset_class=str(final_payload.get('asset_class', 'crypto')),
                    signal_id=uuid.UUID(str(final_payload.get('signal_id'))),
                    symbol=str(final_payload.get('symbol')),
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
