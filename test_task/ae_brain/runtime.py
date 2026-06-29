"""Live runtime wiring: DB + broker + inference engine.

Glues the async pieces together for the production loop:

    data.candidates.ai  -->  InferenceEngine.evaluate  -->  signal.final
                                     |
                                     +--> signal_feature_logs (PostgreSQL)
"""

from __future__ import annotations

import asyncio
import signal as _signal

from ae_brain.config import Settings, get_settings
from ae_brain.contracts import FinalSignal, TradeCandidate
from ae_brain.data.database import Database
from ae_brain.inference.engine import InferenceEngine
from ae_brain.messaging.rabbitmq import SignalBroker
from ae_brain.utils.logging import configure_logging, get_logger

log = get_logger("ae_brain.runtime")


class LiveRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._db = Database(self._s.database)
        self._engine = InferenceEngine(self._s, db=self._db)
        self._broker = SignalBroker(self._s.rabbitmq)
        self._stopping = asyncio.Event()

    async def _handle(self, candidate: TradeCandidate) -> FinalSignal | None:
        signal = await self._engine.evaluate(candidate)
        # Always publish (including SKIP) so downstream has a full audit trail.
        return signal

    async def start(self) -> None:
        configure_logging(self._s.log_level, self._s.log_json)
        log.info("runtime.starting", env=self._s.env)
        await self._db.connect()
        self._engine.load_models()
        await self._broker.connect()

        consume_task = asyncio.create_task(self._broker.consume(self._handle))
        self._install_signal_handlers()
        await self._stopping.wait()

        consume_task.cancel()
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
        await self._broker.close()
        await self._engine.shutdown()
        await self._db.close()

    # Convenience for one-shot evaluation (CLI / API / tests).
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
