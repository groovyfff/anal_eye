"""Live runtime wiring: DB + broker + inference engine."""

from __future__ import annotations

import asyncio
import signal as _signal

from ae_brain.config import Settings, get_settings
from ae_brain.contracts import FinalSignal, TradeCandidate
from ae_brain.data.database import Database
from ae_brain.inference.engine import InferenceEngine
from ae_brain.layers.news_fusion import apply_news_to_signal
from ae_brain.messaging.news_context_store import NewsContextStore
from ae_brain.messaging.news_features import (
    NewsFeaturesCache,
    NewsFeaturesConsumer,
    attach_news_features_to_candidate,
)
from ae_brain.messaging.news_signal_consumer import NewsSignalConsumer
from ae_brain.messaging.rabbitmq import SignalBroker
from ae_brain.utils.logging import configure_logging, get_logger

log = get_logger("ae_brain.runtime")


class LiveRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._db = Database(self._s.database)
        self._engine = InferenceEngine(self._s, db=self._db)
        self._broker = SignalBroker(
            self._s.amqp_input,
            self._s.amqp_output,
            allow_legacy_guest_vhost=self._s.allow_legacy_guest_vhost,
            min_composite_score=self._s.min_composite_score,
            models_loaded=self._engine.is_ready,
            telegram_cfg=self._s.telegram_debug,
            publish_skipped_decisions=self._s.publish_skipped_decisions,
            disable_signal_dedup_in_test_mode=self._s.disable_signal_dedup_in_test_mode,
            allowed_symbols=self._s.allowed_symbol_set,
            min_publish_confidence=self._s.min_publish_confidence,
            only_btc=self._s.only_btc,
        )
        # Optional news-features consumer (RabbitMQ-only, echo-only). Disabled
        # by default so existing deployments are unaffected.
        self._news_cache: NewsFeaturesCache | None = None
        self._news_consumer: NewsFeaturesConsumer | None = None
        self._news_task: asyncio.Task | None = None
        if self._s.enable_news_features:
            self._news_cache = NewsFeaturesCache(max_age_s=self._s.news_features_max_age_s)
            # Reuse the input broker URL unless a dedicated news AMQP url is set.
            news_url = self._s.news_features_amqp_url or self._s.amqp_input.resolved_url
            self._news_consumer = NewsFeaturesConsumer(
                news_url,
                self._news_cache,
                exchange=self._s.news_features_exchange,
                queue=self._s.news_features_queue,
                routing_key=self._s.news_features_routing_key,
            )
        # Optional news.market_signal fusion (OpenRouter path). Bounded, optional;
        # a no-op when the queue is empty. Never breaks live inference.
        self._news_store: NewsContextStore | None = None
        self._news_signal_consumer: NewsSignalConsumer | None = None
        self._news_signal_task: asyncio.Task | None = None
        if self._s.news_signal_enabled:
            self._news_store = NewsContextStore(ttl_s=self._s.news_signal_ttl_sec)
            signal_url = self._s.news_signal_amqp_url or self._s.amqp_input.resolved_url
            self._news_signal_consumer = NewsSignalConsumer(
                signal_url,
                self._news_store,
                exchange=self._s.news_signal_exchange,
                queue=self._s.news_signal_queue,
                routing_key=self._s.news_signal_routing_key,
                min_relevance=self._s.news_min_relevance,
            )
        self._stopping = asyncio.Event()

    async def _handle(self, candidate: TradeCandidate) -> FinalSignal | None:
        # Attach the latest fresh news snapshot (echo-only). No fresh news =>
        # candidate is untouched and scoring proceeds normally.
        if self._news_cache is not None:
            attach_news_features_to_candidate(
                self._news_cache, candidate.meta, candidate.symbol
            )
        signal = await self._engine.evaluate(candidate)
        # Optional news.market_signal fusion: a bounded, capped nudge to
        # confidence/EV. No active news ⇒ signal returned unchanged (identical
        # to the no-news behavior). SKIP is never changed. Best-effort: any
        # failure leaves the original signal intact.
        if self._news_store is not None and signal is not None:
            try:
                agg = self._news_store.aggregate(candidate.symbol)
                signal = apply_news_to_signal(
                    signal,
                    agg,
                    max_conf_delta=self._s.news_max_conf_delta,
                    max_ev_multiplier_delta=self._s.news_max_ev_multiplier_delta,
                )
            except Exception as exc:  # noqa: BLE001 - news must never break inference
                log.warning("runtime.news_fusion_failed", err=str(exc))
        return signal

    async def start(self) -> None:
        configure_logging(self._s.log_level, self._s.log_json)
        log.info("runtime.starting", env=self._s.env, news_features=self._s.enable_news_features)
        log.info(
            "runtime.effective_config",
            allowed_symbols=sorted(self._s.allowed_symbol_set),
            min_publish_confidence=self._s.min_publish_confidence,
            publish_skipped_decisions=self._s.publish_skipped_decisions,
            only_btc=self._s.only_btc,
            model_artifacts_path=str(self._s.model.artifacts_dir.resolve()),
            model_mode=self._s.fusion.meta_mode,
            direct_telegram_enabled=self._s.telegram_debug.enabled,
        )
        await self._db.connect()
        self._engine.load_models()
        if not self._engine.is_ready():
            log.warning("runtime.models_not_ready", msg="inference may SKIP with model_not_loaded")
        await self._broker.connect()

        consume_task = asyncio.create_task(self._broker.consume(self._handle))
        if self._news_consumer is not None:
            try:
                await self._news_consumer.connect()
                self._news_task = asyncio.create_task(self._news_consumer.consume())
                log.info("runtime.news_features_consumer_started")
            except Exception as exc:  # noqa: BLE001 - news is best-effort
                log.warning("runtime.news_features_consumer_failed", err=str(exc))
                self._news_consumer = None
        if self._news_signal_consumer is not None:
            try:
                await self._news_signal_consumer.connect()
                self._news_signal_task = asyncio.create_task(self._news_signal_consumer.consume())
                log.info("runtime.news_signal_consumer_started")
            except Exception as exc:  # noqa: BLE001 - news is best-effort
                log.warning("runtime.news_signal_consumer_failed", err=str(exc))
                self._news_signal_consumer = None
        self._install_signal_handlers()
        await self._stopping.wait()

        consume_task.cancel()
        if self._news_task is not None:
            self._news_task.cancel()
        if self._news_signal_task is not None:
            self._news_signal_task.cancel()
        await self.shutdown()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stopping.set)
            except NotImplementedError:  # pragma: no cover - e.g. Windows
                pass

    async def shutdown(self) -> None:
        log.info("runtime.stopping")
        if self._news_consumer is not None:
            await self._news_consumer.close()
        if self._news_signal_consumer is not None:
            await self._news_signal_consumer.close()
        await self._broker.close()
        await self._engine.shutdown()
        await self._db.close()

    async def evaluate_once(self, candidate: TradeCandidate, use_db: bool = False) -> FinalSignal:
        if use_db:
            await self._db.connect()
        self._engine.load_models()
        try:
            return await self._engine.evaluate(candidate)
        finally:
            await self._engine.shutdown()
            if use_db:
                await self._db.close()


async def run_live(settings: Settings | None = None) -> None:
    await LiveRuntime(settings).start()
