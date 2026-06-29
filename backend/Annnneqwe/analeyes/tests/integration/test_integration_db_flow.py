"""Layer 2 - real-DB hybrid path: backend pre-INSERT -> ae_brain UPDATE.

Boots a real PostgreSQL container, builds the *production backend schema* via the
shared ORM metadata (``OrmBase.metadata.create_all``, the same call the live
``DatabaseManager.initialize`` makes), performs the backend pre-INSERT, then runs
``ae_brain.data.database.Database.update_signal_log`` against that exact row and
asserts the ensemble outputs land in the real ``ai_*`` / ``features_json`` columns
with no ``UndefinedColumnError``.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from ae_brain.config import DatabaseConfig
from ae_brain.contracts import Decision, FinalSignal, LayerProbabilities
from ae_brain.data.database import Database
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from shared.database.signal_log_repository import save_external_candidate_log


def _final_signal_long(symbol: str, signal_id: str) -> FinalSignal:
    """A deterministic LONG decision so the ai_* columns are meaningfully set."""
    return FinalSignal(
        symbol=symbol,
        decision=Decision.LONG,
        position_size_pct=0.12,
        leverage=3.0,
        take_profit=215.0,
        stop_loss=195.0,
        entry_reference=200.0,
        expected_value_usd=42.5,
        confidence=0.71,
        signal_id=signal_id,
        asset_class="stock",
        ev={"is_positive_ev": True, "prob_tp": 0.61, "prob_sl": 0.33},
    )


def test_db_hybrid_pre_insert_then_update(pg_params, candidate_factory):
    async def run() -> None:
        # Build the real backend schema (create_all over the shared ORM metadata).
        await DatabaseManager.initialize(database_url=pg_params["async_url"])
        try:
            payload = candidate_factory(asset_class="stock", n_candles=64)

            # --- Backend pre-INSERT (generates the primary-key row) ----------
            db_id: int | None = None
            async for session in DatabaseManager.get_session():
                db_id = await save_external_candidate_log(session, payload)
            assert isinstance(db_id, int) and db_id > 0

            # --- ae_brain strict UPDATE on that exact row --------------------
            cfg = DatabaseConfig(
                host=pg_params["host"],
                port=pg_params["port"],
                user=pg_params["user"],
                password=pg_params["password"],
                name=pg_params["name"],
            )
            db = Database(cfg)
            await db.connect()
            features = {"vol_z": 0.12, "rsi_14": 55.0, "macd_hist": 0.08}
            probs = LayerProbabilities(
                tabular_p_up=0.70,
                sequence_p_continuation=0.66,
                sequence_trend_sign=1.0,
                rl_target_exposure=0.40,
            )
            signal = _final_signal_long(payload["symbol"], payload["signal_id"])
            try:
                updated = await db.update_signal_log(
                    signal_log_db_id=db_id,
                    features=features,
                    layer_probs=probs.as_dict(),
                    signal=signal,
                    asset_class="stock",
                )
            finally:
                await db.close()

            # Exactly one row affected; no UndefinedColumnError was raised.
            assert updated is True

            # --- Verify the ensemble outputs landed in the real columns ------
            async for session in DatabaseManager.get_session():
                row = (
                    await session.execute(
                        select(SignalFeatureLog).where(SignalFeatureLog.id == db_id)
                    )
                ).scalar_one()

                assert row.ai_signal_type == "LONG"
                assert row.ai_confidence == pytest.approx(0.71)
                assert row.ai_tp_price_suggestion == pytest.approx(215.0)
                assert row.ai_sl_price_suggestion == pytest.approx(195.0)
                assert row.ai_entry_price_suggestion == pytest.approx(200.0)
                assert row.ai_leverage_suggestion == pytest.approx(3.0)
                assert row.ai_consensus_achieved is True
                assert row.ai_reason_summary and "ev_usd=" in row.ai_reason_summary
                # JSONB round-trips back to a dict.
                assert row.features_json == features
                # Pre-INSERT identity columns are preserved by the UPDATE.
                assert row.asset_class == "stock"
                assert str(row.signal_id_uuid) == payload["signal_id"]
        finally:
            await DatabaseManager.close()

    asyncio.run(run())
