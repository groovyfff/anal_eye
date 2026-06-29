from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRACKER_ROOT = ROOT / 'services' / 'tracker-service'
sys.path.insert(0, str(ROOT / 'shared' / 'src'))
sys.path.insert(0, str(TRACKER_ROOT))

from shared.market_hours import MarketHours
from src.logic.external_prices import ExternalPriceStore
from src.logic.signal_tracker import SignalState, SignalTracker, TrackedSignal


@pytest.fixture
def market_hours() -> MarketHours:
    return MarketHours(
        {
            'timezone': 'America/New_York',
            'stock_open': '09:30',
            'stock_close': '16:00',
            'use_fixed_utc_window': True,
            'stock_open_utc': '14:30',
            'stock_close_utc': '21:00',
        }
    )


def test_session_aware_expiration_pauses_when_market_closed(market_hours: MarketHours) -> None:
    """Сигнал перед закрытием не истекает за закрытую сессию — тикают только секунды открытого рынка."""
    store = ExternalPriceStore(max_age_ms=4500)
    tracker = SignalTracker(market_hours=market_hours, price_store=store, expiration_hours=24, db_enabled=False)
    friday_open = dt.datetime(2026, 6, 26, 20, 50, tzinfo=dt.timezone.utc)
    signal = TrackedSignal(
        signal_id='test-signal',
        source_ai='ensemble',
        symbol='AAPL',
        asset_class='stock',
        direction='LONG',
        entry_price='market',
        tp_price=200.0,
        sl_price=180.0,
        leverage=1.0,
        signal_time=friday_open,
        state=SignalState.ENTERED,
        fill_price=190.0,
        actual_entry_time=friday_open,
        last_tick_at=friday_open,
    )
    saturday = dt.datetime(2026, 6, 27, 15, 0, tzinfo=dt.timezone.utc)
    store.upsert_external_message(
        {
            'symbol': 'AAPL',
            'asset_class': 'stock',
            'price': 191.0,
            'bid': 190.9,
            'ask': 191.1,
            'ts': int(saturday.timestamp() * 1000),
        },
        now_utc=saturday,
    )
    new_state, _ = tracker.process_market_data(signal, store.build_market_data_map(now_utc=saturday), now_utc=saturday)
    assert signal.state == SignalState.ENTERED
    assert new_state is None
    assert signal.entered_effective_elapsed_seconds == 0.0
    assert signal.closed_market_seconds > 0

    signal.entered_effective_elapsed_seconds = 24 * 3600 + 1
    new_state, _ = tracker.process_market_data(signal, store.build_market_data_map(now_utc=saturday), now_utc=saturday)
    assert new_state == SignalState.EXPIRED


def test_effective_clock_frozen_across_weekend_then_resumes(market_hours: MarketHours) -> None:
    """Stock: effective_elapsed_seconds стоит на месте всю Fri-close→Mon-open паузу.

    Тикаем в субботу и воскресенье (биржа закрыта): оба клока (общий и
    «после входа») заморожены, растёт только closed_market_seconds. В понедельник
    после открытия счётчик возобновляется.
    """
    store = ExternalPriceStore(max_age_ms=4500)
    # Large expiration so the test isolates the freeze/resume of the clock itself.
    tracker = SignalTracker(market_hours=market_hours, price_store=store, expiration_hours=1000, db_enabled=False)

    friday_close = dt.datetime(2026, 6, 26, 20, 55, tzinfo=dt.timezone.utc)  # Fri, session open
    signal = TrackedSignal(
        signal_id='stock-weekend',
        source_ai='ensemble',
        symbol='AAPL',
        asset_class='stock',
        direction='LONG',
        entry_price='market',
        tp_price=200.0,
        sl_price=180.0,
        leverage=1.0,
        signal_time=friday_close,
        state=SignalState.ENTERED,
        fill_price=190.0,
        actual_entry_time=friday_close,
        last_tick_at=friday_close,
    )

    def _tick(now_utc: dt.datetime):
        store.upsert_external_message(
            {
                'symbol': 'AAPL',
                'asset_class': 'stock',
                'price': 191.0,
                'bid': 190.9,
                'ask': 191.1,
                'ts': int(now_utc.timestamp() * 1000),
            },
            now_utc=now_utc,
        )
        return tracker.process_market_data(signal, store.build_market_data_map(now_utc=now_utc), now_utc=now_utc)

    saturday = dt.datetime(2026, 6, 27, 15, 0, tzinfo=dt.timezone.utc)  # weekend -> closed
    sunday = dt.datetime(2026, 6, 28, 15, 0, tzinfo=dt.timezone.utc)  # weekend -> closed

    _tick(saturday)
    assert signal.state == SignalState.ENTERED
    assert signal.effective_elapsed_seconds == 0.0
    assert signal.entered_effective_elapsed_seconds == 0.0
    assert signal.closed_market_seconds > 0
    closed_after_saturday = signal.closed_market_seconds

    _tick(sunday)
    # Still frozen across the full weekend; only closed-market time accrues.
    assert signal.effective_elapsed_seconds == 0.0
    assert signal.entered_effective_elapsed_seconds == 0.0
    assert signal.closed_market_seconds > closed_after_saturday

    monday_open = dt.datetime(2026, 6, 29, 14, 35, tzinfo=dt.timezone.utc)  # Mon, session open
    _tick(monday_open)
    # Clock resumes once the exchange reopens.
    assert signal.effective_elapsed_seconds > 0.0
    assert signal.entered_effective_elapsed_seconds > 0.0
