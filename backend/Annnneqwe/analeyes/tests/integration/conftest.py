"""Shared fixtures for Layer 2 component-integration tests.

These tests boot *real* PostgreSQL and RabbitMQ containers (via testcontainers)
to exercise the cross-repository contracts under production-like conditions:

* the backend ``shared`` package builds the real ORM schema and pre-INSERTs the
  ``signal_feature_logs`` row, and
* the ``ae_brain`` ensemble writes/serves against that same database / broker.

Run pattern (from ``backend/Annnneqwe/analeyes``)::

    PYTHONPATH="shared/src:services/tracker-service" \
        ../../.venv-test/bin/python -m pytest tests/integration -v
"""

from __future__ import annotations

import os

# Ryuk (the testcontainers resource-reaper) is flaky in sandboxed/CI Docker
# setups; disable it so container lifecycle is managed by the context managers.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

import datetime as dt
import sys
import uuid
from pathlib import Path

import pytest

# Make the backend ``shared`` package importable even if PYTHONPATH was not set.
_ANALEYES_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ANALEYES_ROOT / "shared" / "src", _ANALEYES_ROOT / "services" / "tracker-service"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

from ae_brain.training.synthetic import generate_synthetic_candles

PG_USER = "ae"
PG_PASSWORD = "ae"
PG_DB = "analeyes"


@pytest.fixture(scope="session")
def postgres():
    with PostgresContainer(
        "postgres:16-alpine", username=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    ) as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq():
    with RabbitMqContainer("rabbitmq:3.13-alpine", username="guest", password="guest") as container:
        yield container


@pytest.fixture
def pg_params(postgres) -> dict:
    host = postgres.get_container_host_ip()
    port = int(postgres.get_exposed_port(5432))
    return {
        "host": host,
        "port": port,
        "user": PG_USER,
        "password": PG_PASSWORD,
        "name": PG_DB,
        # SQLAlchemy async URL for the backend DatabaseManager.
        "async_url": f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{host}:{port}/{PG_DB}",
    }


@pytest.fixture
def amqp_url(rabbitmq) -> str:
    host = rabbitmq.get_container_host_ip()
    port = int(rabbitmq.get_exposed_port(5672))
    return f"amqp://guest:guest@{host}:{port}/"


@pytest.fixture
def candidate_factory():
    """Factory producing a backend-shaped ``data.candidates.ai`` payload.

    Mirrors ``ExternalMarketsService._build_candidate_payload``: a 64-row
    ``historical_ohlcv`` window (each row keyed by ``timestamp``), the feature
    map, and a nullable ``signal_log_db_id``.
    """

    def _make(
        *,
        signal_log_db_id: int | None = None,
        n_candles: int = 64,
        asset_class: str = "stock",
        symbol: str = "AAPL",
    ) -> dict:
        candles = generate_synthetic_candles(n=n_candles, seed=42)
        historical_ohlcv = [
            {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "open": float(o),
                "high": float(h),
                "low": float(low),
                "close": float(c),
                "volume": float(v),
            }
            for ts, o, h, low, c, v in zip(
                candles["ts"],
                candles["open"],
                candles["high"],
                candles["low"],
                candles["close"],
                candles["volume"],
            )
        ]
        last_close = historical_ohlcv[-1]["close"]
        now = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0)
        features = {
            "current_price": last_close,
            "rsi": 55.0,
            "macd_hist": 0.12,
            "atr": 1.5,
            "vol_rel": 1.1,
            "sp500_correlation": 0.31,
        }
        return {
            "signal_id": str(uuid.uuid4()),
            "symbol": symbol,
            "name": symbol,
            "asset_class": asset_class,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "trigger_reason": "ema_crossover",
            "trigger_reasons": ["ema_crossover"],
            "heuristic_signal_consensus": "LONG",
            "features": features,
            "indicators": {"consensus": "BULLISH", "consensus_strength": 0.6, "signals": []},
            "patterns": {"consensus": "BULLISH", "consensus_strength": 0.5, "detected_patterns_info": []},
            "historical_snapshots": [],
            "historical_ohlcv": historical_ohlcv,
            "composite_score": 0.72,
            "entry_price_suggestion": "market",
            "signal_log_db_id": signal_log_db_id,
        }

    return _make
