from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

@dataclass(slots=True)
class MarketHoursConfig:
    timezone: str = 'America/New_York'
    stock_open: str = '09:30'
    stock_close: str = '16:00'
    stock_open_utc: str | None = '14:30'
    stock_close_utc: str | None = '21:00'
    use_fixed_utc_window: bool = True
    pre_market_enabled: bool = False
    after_hours_enabled: bool = False
    metal_breaks_utc: list[str] | None = None

class MarketHours:

    def __init__(self, config: dict) -> None:
        normalized = dict(config or {})
        normalized['metal_breaks_utc'] = self._validate_metal_breaks(normalized.get('metal_breaks_utc'))
        self.config = MarketHoursConfig(**normalized)
        self.local_tz = ZoneInfo(self.config.timezone)

    def is_market_open(self, asset_class: str, now_utc: dt.datetime | None=None) -> bool:
        now = now_utc or dt.datetime.now(tz=dt.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)
        if asset_class in {'stock', 'index'}:
            return self._is_stock_session_open(now)
        if asset_class == 'metal':
            return self._is_metal_session_open(now)
        if asset_class == 'forex':
            return self._is_forex_session_open(now)
        return False

    def _is_stock_session_open(self, now_utc: dt.datetime) -> bool:
        local_time = now_utc.astimezone(self.local_tz)
        if local_time.weekday() >= 5:
            return False
        use_fixed_utc_window = bool(self.config.use_fixed_utc_window) and bool(self.config.stock_open_utc) and bool(self.config.stock_close_utc) and (not self.config.pre_market_enabled) and (not self.config.after_hours_enabled)
        if use_fixed_utc_window:
            start_utc = self._parse_hhmm(str(self.config.stock_open_utc))
            end_utc = self._parse_hhmm(str(self.config.stock_close_utc))
            return start_utc <= now_utc.time() < end_utc
        start = self._parse_hhmm(self.config.stock_open)
        end = self._parse_hhmm(self.config.stock_close)
        if self.config.pre_market_enabled:
            start = dt.time(hour=4, minute=0)
        if self.config.after_hours_enabled:
            end = dt.time(hour=20, minute=0)
        return start <= local_time.time() < end

    def _is_metal_session_open(self, now_utc: dt.datetime) -> bool:
        if now_utc.weekday() >= 5:
            return False
        current = now_utc.time()
        if not dt.time(hour=1, minute=0) <= current < dt.time(hour=22, minute=0):
            return False
        for raw_break in self.config.metal_breaks_utc or []:
            start, end = self._parse_range(raw_break)
            if start <= current < end:
                return False
        return True

    @staticmethod
    def _is_forex_session_open(now_utc: dt.datetime) -> bool:
        weekday = now_utc.weekday()
        if weekday in {0, 1, 2, 3}:
            return True
        if weekday == 4:
            return now_utc.time() < dt.time(hour=22, minute=0)
        if weekday == 6:
            return now_utc.time() >= dt.time(hour=22, minute=0)
        return False

    @staticmethod
    def _parse_hhmm(raw: str) -> dt.time:
        hours, minutes = raw.split(':', maxsplit=1)
        return dt.time(hour=int(hours), minute=int(minutes))

    @staticmethod
    def _parse_range(raw: str) -> tuple[dt.time, dt.time]:
        start_raw, end_raw = raw.split('-', maxsplit=1)
        return (MarketHours._parse_hhmm(start_raw.strip()), MarketHours._parse_hhmm(end_raw.strip()))

    @staticmethod
    def _validate_metal_breaks(raw_value: object) -> list[str]:
        if raw_value is None:
            return []
        if not isinstance(raw_value, list):
            raise ValueError("market_hours.metal_breaks_utc must be a list of 'HH:MM-HH:MM' ranges")
        validated: list[str] = []
        for raw in raw_value:
            if not isinstance(raw, str):
                raise ValueError('market_hours.metal_breaks_utc must contain string ranges')
            if '-' not in raw:
                raise ValueError(f"Invalid metal break range '{raw}': expected 'HH:MM-HH:MM'")
            start, end = MarketHours._parse_range(raw)
            if start >= end:
                raise ValueError(f"Invalid metal break range '{raw}': start must be earlier than end for same-day UTC range")
            validated.append(raw)
        return validated
