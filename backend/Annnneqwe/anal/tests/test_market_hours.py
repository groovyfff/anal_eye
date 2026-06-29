import datetime as dt
import pytest
from src.logic.market_hours import MarketHours

def _hours() -> MarketHours:
    return MarketHours({'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'pre_market_enabled': False, 'after_hours_enabled': False})

def test_stock_session_open() -> None:
    market_hours = _hours()
    now = dt.datetime(2026, 1, 5, 15, 0, tzinfo=dt.timezone.utc)
    assert market_hours.is_market_open('stock', now_utc=now)

def test_stock_session_closed_after_hours() -> None:
    market_hours = _hours()
    now = dt.datetime(2026, 1, 5, 23, 0, tzinfo=dt.timezone.utc)
    assert not market_hours.is_market_open('stock', now_utc=now)

def test_stock_uses_timezone_schedule_when_fixed_utc_flag_disabled() -> None:
    market_hours = MarketHours({'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'stock_open_utc': '00:00', 'stock_close_utc': '00:01', 'use_fixed_utc_window': False, 'pre_market_enabled': False, 'after_hours_enabled': False})
    now = dt.datetime(2026, 1, 5, 15, 0, tzinfo=dt.timezone.utc)
    assert market_hours.is_market_open('stock', now_utc=now)

def test_stock_can_use_fixed_utc_window_when_explicitly_enabled() -> None:
    market_hours = MarketHours({'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'stock_open_utc': '14:30', 'stock_close_utc': '21:00', 'use_fixed_utc_window': True, 'pre_market_enabled': False, 'after_hours_enabled': False})
    open_ts = dt.datetime(2026, 1, 5, 15, 0, tzinfo=dt.timezone.utc)
    closed_ts = dt.datetime(2026, 1, 5, 22, 0, tzinfo=dt.timezone.utc)
    assert market_hours.is_market_open('stock', now_utc=open_ts)
    assert not market_hours.is_market_open('stock', now_utc=closed_ts)

def test_metals_session_boundaries() -> None:
    market_hours = _hours()
    open_ts = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.timezone.utc)
    closed_ts = dt.datetime(2026, 1, 5, 23, 0, tzinfo=dt.timezone.utc)
    assert market_hours.is_market_open('metal', now_utc=open_ts)
    assert not market_hours.is_market_open('metal', now_utc=closed_ts)

def test_forex_session() -> None:
    market_hours = _hours()
    sunday_open = dt.datetime(2026, 1, 4, 22, 30, tzinfo=dt.timezone.utc)
    saturday_closed = dt.datetime(2026, 1, 3, 12, 0, tzinfo=dt.timezone.utc)
    assert market_hours.is_market_open('forex', now_utc=sunday_open)
    assert not market_hours.is_market_open('forex', now_utc=saturday_closed)

def test_invalid_metal_break_range_raises() -> None:
    with pytest.raises(ValueError):
        MarketHours({'timezone': 'America/New_York', 'stock_open': '09:30', 'stock_close': '16:00', 'metal_breaks_utc': ['22:00-21:00']})
