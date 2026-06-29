"""Layer 3 - full end-to-end multi-asset flow with session boundaries.

Wires the entire production chain inside the testcontainers environment::

    external-markets payload --> RabbitMQ --> ae_brain (LiveRuntime)
        --> RabbitMQ (signal.final) --> tracker-service (SignalServiceApp)
                                    \\-> notification-service (stubbed Telegram)

What this suite proves
----------------------
* **Deterministic execution** - synthetic layer outputs are injected into the
  ensemble so every candidate resolves to a high-conviction ``LONG`` (instead of
  the default ``SKIP``), exercising every downstream component.
* **Live cross-service communication** - candidates are published over a real
  broker, ``ae_brain`` consumes + UPDATEs the pre-inserted ``signal_feature_logs``
  row + publishes ``signal.final``; the real ``tracker-service`` app consumes it
  off RabbitMQ (``active_tracked_signals``), and the notification consumer routes
  it through the real ``TelegramSender`` topic logic.
* **Session-aware tracking** - the captured ``signal.final`` is replayed through
  a deterministic ``SignalTracker`` with controlled ``now_utc`` to assert the
  session clocks behave per asset class:
    - crypto advances 24/7,
    - stock freezes across the Fri-close -> Mon-open weekend then resumes,
    - forex freezes across the Fri 22:00 -> Sun 22:00 UTC closure,
    - metal freezes inside a configured ``metal_breaks_utc`` window.
* **Boundary check** - a fast, isolated subprocess proves
  ``external-markets-service`` suppresses candidate generation when a traditional
  market is closed.

Run pattern (from ``backend/Annnneqwe/analeyes``)::

    PYTHONPATH="shared/src:services/tracker-service" \
        ../../.venv-test/bin/python -m pytest tests/integration/test_l3_e2e_flow.py -v
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import aio_pika
import asyncpg
import orjson
import pytest
from sqlalchemy import select

from ae_brain.config import Settings
from ae_brain.layers.risk_agent import RiskAgentPrediction
from ae_brain.layers.sequence import SequencePrediction
from ae_brain.layers.tabular import TabularPrediction
from ae_brain.runtime import LiveRuntime
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from shared.database.signal_log_repository import save_external_candidate_log

# tracker-service `src` package (on sys.path via conftest bootstrap).
from src.logic.external_prices import ExternalPriceStore
from src.logic.signal_tracker import SignalState, SignalTracker
from src.main import SignalServiceApp
from shared.market_hours import MarketHours

_ANALEYES_ROOT = Path(__file__).resolve().parents[2]

EXCHANGE = "analeyes_exchange"
CANDIDATE_RK = "data.candidates.ai"
SIGNAL_FINAL_RK = "signal.final"
SIGNAL_FINAL_QUEUE = "q_new_signals_for_tracker"
CAPTURE_QUEUE = "q_l3_capture"
NOTIFICATION_QUEUE = "q_l3_notification"

# notification-service topic routing table (mirrors config/settings.yml shape).
TOPICS = {"crypto": 9001, "stock": 9002, "forex": 9003, "metal": 9004}

# Single market-hours config drives all asset classes:
#   * stock  -> fixed 14:30-21:00 UTC window (weekday gated via NY tz),
#   * metal  -> 01:00-22:00 UTC weekdays minus the break below,
#   * forex  -> static Fri 22:00 -> Sun 22:00 UTC weekend closure.
MH_CONFIG = {
    "timezone": "America/New_York",
    "use_fixed_utc_window": True,
    "stock_open_utc": "14:30",
    "stock_close_utc": "21:00",
    "pre_market_enabled": False,
    "after_hours_enabled": False,
    "metal_breaks_utc": ["13:00-14:00"],
}


def _utc(year, month, day, hour=0, minute=0, second=0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)


def _iso(moment: dt.datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Deterministic ensemble: inject synthetic layer outputs -> high-conviction LONG
# --------------------------------------------------------------------------- #
def _inject_long_weights(engine) -> None:
    """Replace the three layer predictors with deterministic LONG-biased stubs.

    p_up=0.94 / continuation=0.95 (trend +1) / rl_exposure=+0.85 fuse to a
    conviction of ~0.88 (>= min_conviction 0.55) with strongly positive EV, so
    the fusion gate emits LONG rather than SKIP.
    """
    engine._tabular.predict = lambda features: TabularPrediction(p_up=0.94, raw_score=0.94)
    engine._sequence.predict = lambda candles: SequencePrediction(p_continuation=0.95, trend_sign=1.0)
    engine._rl.predict = lambda obs: RiskAgentPrediction(target_exposure=0.85, state_value=0.6)


# --------------------------------------------------------------------------- #
# notification-service: load the real TelegramSender (matplotlib/mplfinance are
# stubbed so the heavy chart deps aren't required; charts aren't built for
# signal.final payloads, which carry no historical_ohlcv).
# --------------------------------------------------------------------------- #
def _load_telegram_sender():
    for name in ("matplotlib", "matplotlib.pyplot", "mplfinance"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["matplotlib"].use = lambda *a, **k: None  # type: ignore[attr-defined]
    path = _ANALEYES_ROOT / "services" / "notification-service" / "src" / "logic" / "telegram_sender.py"
    spec = importlib.util.spec_from_file_location("l3_telegram_sender", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.TelegramSender


async def _seed_asset_correlations(pg_params: dict, pairs: list[tuple[str, str, float]]) -> None:
    """Create + seed the ae_brain ``asset_correlations`` snapshot table.

    Lets ``InferenceEngine._correlated_exposure`` exercise its real fetch/sum
    path (correlations >= threshold) instead of silently returning 0.0.
    """
    conn = await asyncpg.connect(
        host=pg_params["host"],
        port=pg_params["port"],
        user=pg_params["user"],
        password=pg_params["password"],
        database=pg_params["name"],
    )
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_correlations (
                id SERIAL PRIMARY KEY,
                base_symbol TEXT NOT NULL,
                quote_symbol TEXT NOT NULL,
                correlation DOUBLE PRECISION NOT NULL,
                window_candles INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute("TRUNCATE asset_correlations")
        for base, quote, corr in pairs:
            await conn.execute(
                "INSERT INTO asset_correlations (base_symbol, quote_symbol, correlation, window_candles)"
                " VALUES ($1, $2, $3, 200)",
                base,
                quote,
                corr,
            )
    finally:
        await conn.close()


def _ae_brain_settings(pg_params: dict, amqp_url: str) -> Settings:
    settings = Settings()
    settings.executor.process_workers = 0  # thread pool only (no pickling)
    settings.rabbitmq.url = amqp_url
    settings.rabbitmq.publish_exchange = EXCHANGE  # topic exchange -> tracker/notif bindings
    settings.rabbitmq.consume_queue = CANDIDATE_RK
    settings.rabbitmq.publish_routing_key = SIGNAL_FINAL_RK
    settings.database.host = pg_params["host"]
    settings.database.port = pg_params["port"]
    settings.database.user = pg_params["user"]
    settings.database.password = pg_params["password"]
    settings.database.name = pg_params["name"]
    return settings


def _tracker_settings(amqp_url: str) -> dict:
    return {
        "rabbitmq": {
            "url": amqp_url,
            "exchange": EXCHANGE,
            "signal_final_queue": SIGNAL_FINAL_QUEUE,
            "live_prices_queue": "data.live_prices.external",
            "connect_retries": 10,
            "connect_retry_delay_s": 1.0,
        },
        "signal_tracker": {"check_interval_s": 0.5, "entry_timeout_sec": 300, "expiration_hours": 24},
        "market_hours": MH_CONFIG,
        "database": {"enabled": False},  # tracker DB path covered in Layer 2
        "prometheus": {"enabled": False},
        "trading": {"default_initial_bank_usd": 500},
        "logging": {"service_name": "tracker-service", "level": "ERROR"},
    }


async def _capture_signal_final(channel, signal_id: str, timeout: float = 45.0) -> dict | None:
    queue = await channel.declare_queue(CAPTURE_QUEUE, durable=True)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        msg = await queue.get(no_ack=True, fail=False)
        if msg is not None:
            data = orjson.loads(msg.body)
            if data.get("signal_id") == signal_id:
                return data
            continue  # different scenario's final - ignore
        await asyncio.sleep(0.2)
    return None


# --------------------------------------------------------------------------- #
# Deterministic session-clock driver (replays a captured signal.final through a
# standalone SignalTracker with controlled now_utc).
# --------------------------------------------------------------------------- #
def _standalone_tracker() -> SignalTracker:
    return SignalTracker(
        market_hours=MarketHours(MH_CONFIG),
        price_store=ExternalPriceStore(max_age_ms=10**12),
        entry_timeout_sec=300,
        expiration_hours=100_000,  # avoid EXPIRE across long simulated gaps
        slippage_pct=0.001,
        db_enabled=False,
    )


def _start_tracking(final: dict, *, symbol: str, signal_time: dt.datetime):
    payload = dict(final)
    payload["symbol"] = symbol
    payload["signal_time"] = _iso(signal_time)
    tracker = _standalone_tracker()
    tracked = tracker.start_tracking_signal(payload)
    assert tracked is not None, "high-conviction signal must be accepted (not SKIP)"
    return tracker, tracked


def _tick(tracker, tracked, price: float, now_utc: dt.datetime):
    return tracker.process_market_data(tracked, {tracked.symbol: {"price": price}}, now_utc=now_utc)


# =========================================================================== #
# Main end-to-end test
# =========================================================================== #
# The tracker-service's blocking pika consumer runs in a daemon thread and, on
# teardown, its in-flight ``process_data_events`` can observe the TCP EOF as a
# ``StreamLostError`` (benign shutdown race in BlockingConnection). It does not
# affect the assertions, so we ignore that thread-exception warning here.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_l3_end_to_end_multi_asset(pg_params, amqp_url, candidate_factory):
    assets = [
        {"asset_class": "crypto", "symbol": "BTCUSDT"},
        {"asset_class": "stock", "symbol": "AAPL"},
        {"asset_class": "forex", "symbol": "EURUSD"},
        {"asset_class": "metal", "symbol": "XAUUSD"},
    ]

    async def run() -> None:
        # 1) Real backend schema + correlation snapshot seed.
        await DatabaseManager.initialize(database_url=pg_params["async_url"])
        await _seed_asset_correlations(
            pg_params,
            [
                ("BTCUSDT", "ETHUSDT", 0.72),
                ("AAPL", "MSFT", 0.70),
                ("EURUSD", "GBPUSD", 0.68),
                ("XAUUSD", "SI=F", 0.66),
            ],
        )

        # 2) ae_brain runtime with deterministic LONG ensemble.
        runtime = LiveRuntime(_ae_brain_settings(pg_params, amqp_url))
        await runtime._db.connect()
        runtime._engine.load_models()
        _inject_long_weights(runtime._engine)
        await runtime._broker.connect()
        consume_task = asyncio.create_task(runtime._broker.consume(runtime._handle))

        # 3) Broker topology + notification consumer (real TelegramSender routing).
        conn = await aio_pika.connect_robust(amqp_url)
        channel = await conn.channel()
        exchange = await channel.declare_exchange(EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
        capture_q = await channel.declare_queue(CAPTURE_QUEUE, durable=True)
        await capture_q.bind(exchange, SIGNAL_FINAL_RK)
        notif_q = await channel.declare_queue(NOTIFICATION_QUEUE, durable=True)
        await notif_q.bind(exchange, SIGNAL_FINAL_RK)

        telegram_sender = _load_telegram_sender()({"telegram": {"asset_class_topics": TOPICS}})
        notifications: list[dict] = []

        async def _on_notification(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with message.process():
                payload = orjson.loads(message.body)
                topic_id = telegram_sender.resolve_topic_id(
                    payload.get("asset_class"), payload.get("source_ai")
                )
                await telegram_sender.send_signal(payload)  # stdout stub (no bot token)
                notifications.append(
                    {
                        "topic_id": topic_id,
                        "asset_class": payload.get("asset_class"),
                        "symbol": payload.get("symbol"),
                        "decision": payload.get("decision"),
                        "text": telegram_sender.format_signal_message(payload),
                    }
                )

        await notif_q.consume(_on_notification)

        # 4) Boot the real tracker-service app (consumes signal.final over RabbitMQ).
        tracker_app = SignalServiceApp(_tracker_settings(amqp_url))
        tracker_task = asyncio.create_task(tracker_app.run())
        # Wait until the tracker's RabbitMQ consumer is up before publishing.
        for _ in range(60):
            if tracker_app._consumer is not None:
                break
            await asyncio.sleep(0.25)
        assert tracker_app._consumer is not None, "tracker-service did not connect to RabbitMQ"
        await asyncio.sleep(1.0)  # let the consumer thread bind its queues

        finals: dict[str, dict] = {}
        try:
            # 5) Drive each asset class candidate through the live pipeline.
            for asset in assets:
                payload = candidate_factory(
                    asset_class=asset["asset_class"], symbol=asset["symbol"], n_candles=64
                )
                db_id: int | None = None
                async for session in DatabaseManager.get_session():
                    db_id = await save_external_candidate_log(session, payload)
                assert db_id and db_id > 0
                payload["signal_log_db_id"] = db_id

                await channel.default_exchange.publish(
                    aio_pika.Message(body=orjson.dumps(payload), content_type="application/json"),
                    routing_key=CANDIDATE_RK,
                )

                final = await _capture_signal_final(channel, payload["signal_id"], timeout=45.0)
                assert final is not None, f"no signal.final captured for {asset['symbol']}"
                for key in ("tp", "sl", "entry_price", "signal_id", "source_ai", "decision",
                            "asset_class", "signal_log_db_id", "signal_time", "leverage"):
                    assert key in final, f"{asset['symbol']}: missing tracker key {key}"
                assert final["decision"] == "LONG", f"{asset['symbol']} expected LONG, got {final['decision']}"
                assert final["source_ai"] == "ensemble"
                assert final["asset_class"] == asset["asset_class"]
                assert final["signal_log_db_id"] == db_id
                entry, tp, sl = float(final["entry_price"]), float(final["tp"]), float(final["sl"])
                assert tp > entry > sl, f"{asset['symbol']}: bad LONG TP/SL ordering"
                finals[asset["asset_class"]] = final

            # 6) ae_brain performed the strict UPDATE on the pre-inserted row.
            stock_db_id = finals["stock"]["signal_log_db_id"]
            async for session in DatabaseManager.get_session():
                row = (
                    await session.execute(select(SignalFeatureLog).where(SignalFeatureLog.id == stock_db_id))
                ).scalar_one()
                assert row.ai_signal_type == "LONG"
                assert row.ai_confidence is not None
                assert row.features_json is not None

            # 7) The live tracker-service consumed crypto signal.final off RabbitMQ.
            crypto_key = f"{finals['crypto']['signal_id']}-ensemble"
            for _ in range(80):
                if crypto_key in tracker_app.tracker.active_tracked_signals:
                    break
                await asyncio.sleep(0.25)
            assert crypto_key in tracker_app.tracker.active_tracked_signals, (
                "tracker-service did not start tracking the crypto signal.final"
            )

            # 8) Notification routing: every asset routed, stock -> dedicated topic.
            for _ in range(40):
                if len(notifications) >= len(assets):
                    break
                await asyncio.sleep(0.25)
            routed = {n["asset_class"]: n for n in notifications}
            assert "stock" in routed, "notification-service never received the stock signal"
            assert routed["stock"]["topic_id"] == TOPICS["stock"], "stock not routed to its topic"
            assert routed["crypto"]["topic_id"] == TOPICS["crypto"]
            assert routed["forex"]["topic_id"] == TOPICS["forex"]
            assert routed["metal"]["topic_id"] == TOPICS["metal"]

            # 9) Deterministic session-clock assertions per asset class.
            _assert_crypto_runs_247(finals["crypto"])
            _assert_stock_weekend_freeze(finals["stock"])
            _assert_forex_weekend_halt(finals["forex"])
            _assert_metal_break_freeze(finals["metal"])
        finally:
            await tracker_app.stop()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(tracker_task, timeout=15.0)
            consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consume_task
            await conn.close()
            await runtime.shutdown()
            await DatabaseManager.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Per-asset session-boundary scenarios
# --------------------------------------------------------------------------- #
def _assert_crypto_runs_247(final: dict) -> None:
    """Crypto: duration timers advance at *any* timestamp (incl. weekends)."""
    tracker, sig = _start_tracking(final, symbol="BTCUSDT", signal_time=_utc(2026, 6, 27, 12, 0, 0))
    price = float(sig.entry_price)

    # Saturday: enter immediately, clocks already advancing.
    state, _ = _tick(tracker, sig, price, _utc(2026, 6, 27, 12, 0, 30))
    assert state == SignalState.ENTERED
    eff_after_entry = sig.effective_elapsed_seconds
    assert eff_after_entry > 0.0  # weekend time counts for crypto

    # +1h still Saturday -> both effective + entered clocks keep moving.
    _tick(tracker, sig, price, _utc(2026, 6, 27, 13, 0, 30))
    assert sig.effective_elapsed_seconds > eff_after_entry
    assert sig.entered_effective_elapsed_seconds > 0.0

    # Sunday -> still advancing, never frozen.
    eff_sat = sig.effective_elapsed_seconds
    _tick(tracker, sig, price, _utc(2026, 6, 28, 12, 0, 0))
    assert sig.effective_elapsed_seconds > eff_sat
    assert sig.closed_market_seconds == 0.0  # crypto is never "closed"
    assert sig.state == SignalState.ENTERED


def _assert_stock_weekend_freeze(final: dict) -> None:
    """Stock: effective clocks freeze over Fri-close -> Mon-open, then resume."""
    tracker, sig = _start_tracking(final, symbol="AAPL", signal_time=_utc(2026, 6, 26, 15, 0, 0))
    price = float(sig.entry_price)

    # Friday inside the 14:30-21:00 UTC window -> ENTERED.
    state, _ = _tick(tracker, sig, price, _utc(2026, 6, 26, 15, 0, 30))
    assert state == SignalState.ENTERED
    _tick(tracker, sig, price, _utc(2026, 6, 26, 20, 0, 0))
    eff_friday = sig.effective_elapsed_seconds
    entered_friday = sig.entered_effective_elapsed_seconds
    assert eff_friday > 0.0 and entered_friday > 0.0

    # Saturday + Sunday: market closed -> effective + entered clocks frozen.
    _tick(tracker, sig, price, _utc(2026, 6, 27, 12, 0, 0))
    assert sig.effective_elapsed_seconds == eff_friday
    assert sig.entered_effective_elapsed_seconds == entered_friday
    closed_after_sat = sig.closed_market_seconds
    assert closed_after_sat > 0.0

    _tick(tracker, sig, price, _utc(2026, 6, 28, 12, 0, 0))
    assert sig.effective_elapsed_seconds == eff_friday
    assert sig.entered_effective_elapsed_seconds == entered_friday
    assert sig.closed_market_seconds > closed_after_sat

    # Monday open -> clocks resume.
    _tick(tracker, sig, price, _utc(2026, 6, 29, 15, 0, 0))
    assert sig.effective_elapsed_seconds > eff_friday
    assert sig.entered_effective_elapsed_seconds > entered_friday


def _assert_forex_weekend_halt(final: dict) -> None:
    """Forex: Fri 22:00 -> Sun 22:00 UTC closure halts entry/expiry clocks."""
    tracker, sig = _start_tracking(final, symbol="EURUSD", signal_time=_utc(2026, 6, 26, 20, 0, 0))
    price = float(sig.entry_price)

    state, _ = _tick(tracker, sig, price, _utc(2026, 6, 26, 20, 0, 30))  # Fri < 22:00 -> open
    assert state == SignalState.ENTERED
    _tick(tracker, sig, price, _utc(2026, 6, 26, 21, 0, 0))
    eff_open = sig.effective_elapsed_seconds
    entered_open = sig.entered_effective_elapsed_seconds
    assert eff_open > 0.0 and entered_open > 0.0

    # Saturday -> frozen.
    _tick(tracker, sig, price, _utc(2026, 6, 27, 12, 0, 0))
    assert sig.effective_elapsed_seconds == eff_open
    assert sig.entered_effective_elapsed_seconds == entered_open

    # Sunday 21:00 (< 22:00) -> still frozen.
    _tick(tracker, sig, price, _utc(2026, 6, 28, 21, 0, 0))
    assert sig.effective_elapsed_seconds == eff_open
    assert sig.entered_effective_elapsed_seconds == entered_open
    assert sig.closed_market_seconds > 0.0

    # Sunday 22:30 (>= 22:00) -> forex reopens, clocks resume.
    _tick(tracker, sig, price, _utc(2026, 6, 28, 22, 30, 0))
    assert sig.effective_elapsed_seconds > eff_open
    assert sig.entered_effective_elapsed_seconds > entered_open


def _assert_metal_break_freeze(final: dict) -> None:
    """Metal: a configured intraday break window freezes the tracking clocks."""
    tracker, sig = _start_tracking(final, symbol="XAUUSD", signal_time=_utc(2026, 6, 26, 12, 0, 0))
    price = float(sig.entry_price)

    state, _ = _tick(tracker, sig, price, _utc(2026, 6, 26, 12, 0, 30))  # open (not in break)
    assert state == SignalState.ENTERED
    _tick(tracker, sig, price, _utc(2026, 6, 26, 12, 30, 0))
    eff_open = sig.effective_elapsed_seconds
    entered_open = sig.entered_effective_elapsed_seconds
    assert eff_open > 0.0 and entered_open > 0.0

    # 13:30 UTC falls inside metal_breaks_utc ['13:00-14:00'] -> frozen.
    _tick(tracker, sig, price, _utc(2026, 6, 26, 13, 30, 0))
    assert sig.effective_elapsed_seconds == eff_open
    assert sig.entered_effective_elapsed_seconds == entered_open
    assert sig.closed_market_seconds > 0.0

    # 14:30 UTC back inside the open session -> clocks resume.
    _tick(tracker, sig, price, _utc(2026, 6, 26, 14, 30, 0))
    assert sig.effective_elapsed_seconds > eff_open
    assert sig.entered_effective_elapsed_seconds > entered_open


# =========================================================================== #
# Boundary check (isolated subprocess to avoid src-package collision)
# =========================================================================== #
def test_boundary_closed_market_suppresses_candidate():
    """external-markets-service must not emit candidates while a market is closed."""
    probe = Path(__file__).resolve().parent / "_external_markets_boundary.py"
    env_pythonpath = f"{_ANALEYES_ROOT / 'shared' / 'src'}:{_ANALEYES_ROOT / 'services' / 'external-markets-service'}"
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(_ANALEYES_ROOT),
        env={"PYTHONPATH": env_pythonpath, "PATH": __import__("os").environ.get("PATH", "")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "BOUNDARY_OK" in result.stdout, (
        f"boundary probe failed (rc={result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
