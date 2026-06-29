"""Optional FastAPI control/inference surface.

Provides a thin HTTP interface for health checks and synchronous one-off signal
evaluation (handy for debugging / dashboards). The production decision path is
the RabbitMQ consumer in :mod:`ae_brain.runtime`, not this API.
"""

from __future__ import annotations

from typing import Any

from ae_brain.config import get_settings
from ae_brain.contracts import TradeCandidate
from ae_brain.inference.engine import InferenceEngine


def create_app() -> Any:
    from fastapi import FastAPI
    from pydantic import BaseModel

    settings = get_settings()
    engine = InferenceEngine(settings, db=None)

    app = FastAPI(title="A.E. Brain", version="0.1.0")

    class CandidateIn(BaseModel):
        symbol: str
        interval: str = "5m"
        candles: list[dict]
        # 0 => no backend row; the engine will INSERT (local/dev fallback).
        signal_log_db_id: int = 0
        asset_class: str = "crypto"
        correlation_id: str = ""
        meta: dict = {}

    @app.on_event("startup")
    def _startup() -> None:
        engine.load_models()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": "0.1.0", "feature_dim": engine.expected_feature_dim()}

    @app.post("/evaluate")
    async def evaluate(body: CandidateIn) -> dict:
        candidate = TradeCandidate(
            symbol=body.symbol,
            interval=body.interval,
            candles=body.candles,
            signal_log_db_id=body.signal_log_db_id,
            asset_class=body.asset_class,
            correlation_id=body.correlation_id,
            meta=body.meta,
        )
        signal = await engine.evaluate(candidate)
        return signal.to_dict()

    return app
