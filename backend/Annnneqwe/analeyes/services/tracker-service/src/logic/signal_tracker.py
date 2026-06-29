from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Awaitable

from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from shared.market_hours import MarketHours
from sqlalchemy import select

from src.logic.external_prices import EXTERNAL_ASSET_CLASSES, ExternalPriceStore

logger = logging.getLogger(__name__)

AncillaryCallback = Callable[..., Awaitable[None] | None]


class SignalState(str, Enum):
    WAITING_FOR_ENTRY = 'WAITING_FOR_ENTRY'
    ENTERED = 'ENTERED'
    TP_HIT = 'TP_HIT'
    SL_HIT = 'SL_HIT'
    EXPIRED = 'EXPIRED'
    CANCELLED_UNFILLED = 'CANCELLED_UNFILLED'


TERMINAL_STATES = frozenset(
    {
        SignalState.TP_HIT,
        SignalState.SL_HIT,
        SignalState.EXPIRED,
        SignalState.CANCELLED_UNFILLED,
    }
)


@dataclass
class TrackedSignal:
    signal_id: str
    source_ai: str
    symbol: str
    asset_class: str
    direction: str
    entry_price: float | str
    tp_price: float
    sl_price: float
    leverage: float
    signal_time: dt.datetime
    signal_log_db_id: int | None = None
    slippage_pct: float = 0.001
    max_age_ms: int = 4500
    prefer: str = 'tp'
    state: SignalState = SignalState.WAITING_FOR_ENTRY
    fill_price: float | None = None
    actual_entry_time: dt.datetime | None = None
    hit_price: float | None = None
    hit_time: dt.datetime | None = None
    duration_seconds: int | None = None
    entry_notified: bool = False
    effective_elapsed_seconds: float = 0.0
    entered_effective_elapsed_seconds: float = 0.0
    closed_market_seconds: float = 0.0
    last_tick_at: dt.datetime | None = None

    @property
    def dedup_key(self) -> str:
        return f'{self.signal_id}-{self.source_ai}'

    @property
    def entry_timeout_sec(self) -> int:
        return 300

    @property
    def expiration_sec(self) -> int:
        return 24 * 3600


class SignalTracker:
    """FSM отслеживания сигналов с учётом торговых сессий."""

    def __init__(
        self,
        market_hours: MarketHours,
        price_store: ExternalPriceStore,
        *,
        entry_timeout_sec: int = 300,
        expiration_hours: int = 24,
        slippage_pct: float = 0.001,
        max_age_ms: int = 4500,
        prefer: str = 'tp',
        default_bank_usd: float = 500.0,
        db_enabled: bool = True,
        on_entry: AncillaryCallback | None = None,
        on_outcome: AncillaryCallback | None = None,
        on_duplicate: Callable[[], None] | None = None,
    ) -> None:
        self.market_hours = market_hours
        self.price_store = price_store
        self.entry_timeout_sec = entry_timeout_sec
        self.expiration_hours = expiration_hours
        self.slippage_pct = slippage_pct
        self.max_age_ms = max_age_ms
        self.prefer = prefer.lower()
        self.default_bank_usd = default_bank_usd
        self.db_enabled = db_enabled
        self.on_entry = on_entry
        self.on_outcome = on_outcome
        self.on_duplicate = on_duplicate
        self.active_tracked_signals: dict[str, TrackedSignal] = {}

    @staticmethod
    def build_dedup_key(signal_id: str, source_ai: str) -> str:
        return f'{signal_id}-{source_ai}'

    def start_tracking_signal(self, payload: dict[str, Any]) -> TrackedSignal | None:
        decision = str(payload.get('decision', '')).upper()
        if decision in {'SKIP', 'NONE', ''}:
            logger.info('[tracker] Пропуск сигнала decision=%s symbol=%s', decision, payload.get('symbol'))
            return None
        signal_id = str(payload['signal_id'])
        source_ai = str(payload.get('source_ai', 'unknown'))
        dedup_key = self.build_dedup_key(signal_id, source_ai)
        if dedup_key in self.active_tracked_signals:
            logger.warning('[tracker] Дубликат сигнала key=%s', dedup_key)
            if self.on_duplicate:
                self.on_duplicate()
            return None
        symbol = str(payload['symbol'])
        asset_class = str(payload.get('asset_class', 'crypto')).lower()
        direction = decision if decision in {'LONG', 'SHORT'} else str(payload.get('direction', 'LONG')).upper()
        signal_time = self._parse_dt(payload.get('signal_time') or payload.get('timestamp')) or dt.datetime.now(tz=dt.timezone.utc)
        entry_raw = payload.get('entry_price', payload.get('entry', 'market'))
        entry_price: float | str
        if isinstance(entry_raw, str) and entry_raw.lower() == 'market':
            entry_price = 'market'
        else:
            entry_price = float(entry_raw)
        tp_price = float(payload['tp'])
        sl_price = float(payload['sl'])
        leverage = float(payload.get('leverage', 1.0))
        tracked = TrackedSignal(
            signal_id=signal_id,
            source_ai=source_ai,
            symbol=symbol,
            asset_class=asset_class,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            leverage=leverage,
            signal_time=signal_time,
            signal_log_db_id=payload.get('signal_log_db_id'),
            slippage_pct=self.slippage_pct,
            max_age_ms=self.max_age_ms,
            prefer=self.prefer,
            last_tick_at=signal_time,
        )
        self._validate_tp_sl(tracked)
        self.active_tracked_signals[dedup_key] = tracked
        logger.info(
            '[tracker] Старт отслеживания key=%s symbol=%s direction=%s asset_class=%s',
            dedup_key,
            symbol,
            direction,
            asset_class,
        )
        return tracked

    def _validate_tp_sl(self, signal: TrackedSignal) -> None:
        if signal.entry_price == 'market':
            return
        entry = float(signal.entry_price)
        if signal.direction == 'LONG':
            if signal.tp_price <= entry or signal.sl_price >= entry:
                logger.warning('[tracker] Некорректные TP/SL для LONG symbol=%s', signal.symbol)
        elif signal.direction == 'SHORT':
            if signal.tp_price >= entry or signal.sl_price <= entry:
                logger.warning('[tracker] Некорректные TP/SL для SHORT symbol=%s', signal.symbol)

    @staticmethod
    def _parse_dt(value: Any) -> dt.datetime | None:
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=dt.timezone.utc)
            return value.astimezone(dt.timezone.utc)
        if isinstance(value, str):
            parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        return None

    def _market_time_counts(self, asset_class: str, now_utc: dt.datetime) -> bool:
        if asset_class == 'crypto':
            return True
        return self.market_hours.is_market_open(asset_class, now_utc=now_utc)

    def _advance_session_clocks(self, signal: TrackedSignal, now_utc: dt.datetime) -> None:
        if signal.last_tick_at is None:
            signal.last_tick_at = now_utc
            return
        delta = (now_utc - signal.last_tick_at).total_seconds()
        if delta <= 0:
            return
        if self._market_time_counts(signal.asset_class, now_utc):
            signal.effective_elapsed_seconds += delta
            if signal.state == SignalState.ENTERED:
                signal.entered_effective_elapsed_seconds += delta
        else:
            signal.closed_market_seconds += delta
        signal.last_tick_at = now_utc

    def effective_now(self, signal: TrackedSignal) -> dt.datetime:
        """Виртуальное «сейчас»: signal_time + только секунды открытой сессии."""
        return signal.signal_time + dt.timedelta(seconds=signal.effective_elapsed_seconds)

    def effective_entry_now(self, signal: TrackedSignal) -> dt.datetime | None:
        if signal.actual_entry_time is None:
            return None
        return signal.actual_entry_time + dt.timedelta(seconds=signal.entered_effective_elapsed_seconds)

    def get_current_price(self, signal: TrackedSignal, market_data_map: dict[str, dict[str, Any]], now_utc: dt.datetime) -> float | None:
        pair = market_data_map.get(signal.symbol)
        if pair is not None:
            for key in ('markPrice', 'price', 'lastPrice'):
                raw = pair.get(key)
                if raw is not None:
                    return float(raw)
        if signal.asset_class in EXTERNAL_ASSET_CLASSES:
            quote = self.price_store.get_quote(signal.symbol, signal.asset_class, now_utc=now_utc)
            if quote is not None:
                return quote.price
            return None
        quote = self.price_store.get_quote(signal.symbol, 'crypto', now_utc=now_utc)
        return quote.price if quote else None

    def process_market_data(
        self,
        signal: TrackedSignal,
        market_data_map: dict[str, dict[str, Any]],
        now_utc: dt.datetime | None = None,
    ) -> tuple[SignalState | None, float | None]:
        now = now_utc or dt.datetime.now(tz=dt.timezone.utc)
        if signal.state in TERMINAL_STATES:
            return None, None
        self._advance_session_clocks(signal, now)
        current_price = self.get_current_price(signal, market_data_map, now)
        if current_price is None:
            return None, None
        if signal.state == SignalState.WAITING_FOR_ENTRY:
            return self._process_waiting(signal, current_price, now)
        if signal.state == SignalState.ENTERED:
            return self._process_entered(signal, current_price, now)
        return None, current_price

    def _process_waiting(
        self,
        signal: TrackedSignal,
        current_price: float,
        now_utc: dt.datetime,
    ) -> tuple[SignalState | None, float | None]:
        if signal.effective_elapsed_seconds >= self.entry_timeout_sec:
            signal.state = SignalState.CANCELLED_UNFILLED
            signal.hit_time = now_utc
            logger.info('[tracker] CANCELLED_UNFILLED key=%s (entry timeout)', signal.dedup_key)
            return signal.state, current_price
        if signal.entry_price == 'market':
            signal.fill_price = current_price
            signal.actual_entry_time = now_utc
            signal.state = SignalState.ENTERED
            signal.entered_effective_elapsed_seconds = 0.0
            logger.info('[tracker] ENTERED (market) key=%s fill=%s', signal.dedup_key, current_price)
            return signal.state, current_price
        entry = float(signal.entry_price)
        if signal.direction == 'LONG':
            threshold = entry * (1.0 + signal.slippage_pct)
            if current_price <= threshold:
                signal.fill_price = current_price
                signal.actual_entry_time = now_utc
                signal.state = SignalState.ENTERED
                signal.entered_effective_elapsed_seconds = 0.0
                logger.info('[tracker] ENTERED LONG key=%s fill=%s', signal.dedup_key, current_price)
                return signal.state, current_price
        else:
            threshold = entry * (1.0 - signal.slippage_pct)
            if current_price >= threshold:
                signal.fill_price = current_price
                signal.actual_entry_time = now_utc
                signal.state = SignalState.ENTERED
                signal.entered_effective_elapsed_seconds = 0.0
                logger.info('[tracker] ENTERED SHORT key=%s fill=%s', signal.dedup_key, current_price)
                return signal.state, current_price
        return None, current_price

    def _process_entered(
        self,
        signal: TrackedSignal,
        current_price: float,
        now_utc: dt.datetime,
    ) -> tuple[SignalState | None, float | None]:
        if signal.entered_effective_elapsed_seconds >= self.expiration_hours * 3600:
            signal.state = SignalState.EXPIRED
            signal.hit_price = current_price
            signal.hit_time = now_utc
            logger.info('[tracker] EXPIRED key=%s', signal.dedup_key)
            return signal.state, current_price
        tp_crossed = sl_crossed = False
        if signal.direction == 'LONG':
            tp_crossed = current_price >= signal.tp_price
            sl_crossed = current_price <= signal.sl_price
        else:
            tp_crossed = current_price <= signal.tp_price
            sl_crossed = current_price >= signal.sl_price
        if tp_crossed and sl_crossed:
            logger.warning('[tracker] TP/SL collision key=%s prefer=%s', signal.dedup_key, signal.prefer)
            if signal.prefer == 'sl':
                tp_crossed = False
            else:
                sl_crossed = False
        if tp_crossed:
            signal.state = SignalState.TP_HIT
            signal.hit_price = current_price
            signal.hit_time = now_utc
            logger.info('[tracker] TP_HIT key=%s price=%s', signal.dedup_key, current_price)
            return signal.state, current_price
        if sl_crossed:
            signal.state = SignalState.SL_HIT
            signal.hit_price = current_price
            signal.hit_time = now_utc
            logger.info('[tracker] SL_HIT key=%s price=%s', signal.dedup_key, current_price)
            return signal.state, current_price
        return None, current_price

    def calculate_pnl(self, signal: TrackedSignal) -> tuple[float | None, float | None]:
        if signal.fill_price is None or signal.hit_price is None:
            return None, None
        entry = signal.fill_price
        exit_price = signal.hit_price
        if entry == 0:
            return None, None
        if signal.direction == 'LONG':
            move = (exit_price - entry) / entry
        else:
            move = (entry - exit_price) / entry
        pnl_percent = move * signal.leverage * 100.0
        pnl_usdt = move * self.default_bank_usd * signal.leverage
        return round(pnl_usdt, 4), round(pnl_percent, 4)

    def _compute_duration_seconds(self, signal: TrackedSignal) -> int:
        if signal.actual_entry_time is None or signal.hit_time is None:
            return int(signal.effective_elapsed_seconds)
        return max(0, int(signal.entered_effective_elapsed_seconds))

    async def check_tracked_signals(self, market_data_map: dict[str, dict[str, Any]]) -> None:
        now_utc = dt.datetime.now(tz=dt.timezone.utc)
        for key in list(self.active_tracked_signals.keys()):
            signal = self.active_tracked_signals[key]
            previous_state = signal.state
            new_state, _price = self.process_market_data(signal, market_data_map, now_utc=now_utc)
            if new_state == SignalState.ENTERED and previous_state == SignalState.WAITING_FOR_ENTRY:
                if not signal.entry_notified:
                    signal.entry_notified = True
                    if self.on_entry:
                        await self._maybe_await(self.on_entry, signal)
            if new_state in TERMINAL_STATES:
                signal.duration_seconds = self._compute_duration_seconds(signal)
                await self._handle_signal_completion(signal)

    async def _handle_signal_completion(self, signal: TrackedSignal) -> None:
        pnl_usdt, pnl_percent = self.calculate_pnl(signal)
        if signal.state != SignalState.CANCELLED_UNFILLED:
            await self._update_signal_log_in_db(signal, pnl_usdt, pnl_percent)
        if self.on_outcome:
            await self._maybe_await(self.on_outcome, signal, pnl_usdt, pnl_percent)
        self.active_tracked_signals.pop(signal.dedup_key, None)
        logger.info('[tracker] Завершён сигнал key=%s status=%s', signal.dedup_key, signal.state.value)

    async def _update_signal_log_in_db(
        self,
        signal: TrackedSignal,
        pnl_usdt: float | None,
        pnl_percent: float | None,
    ) -> None:
        if not self.db_enabled or not signal.signal_log_db_id:
            return
        try:
            async for session in DatabaseManager.get_session():
                result = await session.execute(
                    select(SignalFeatureLog).where(SignalFeatureLog.id == int(signal.signal_log_db_id))
                )
                row = result.scalar_one_or_none()
                if row is None:
                    logger.warning('[tracker] signal_feature_logs id=%s не найден', signal.signal_log_db_id)
                    return
                row.tracker_status = signal.state.value
                row.tracker_entry_price = signal.fill_price
                row.tracker_exit_price = signal.hit_price
                row.tracker_pnl_percent = pnl_percent
                row.tracker_pnl_usdt = pnl_usdt
                row.tracker_duration_seconds = signal.duration_seconds
                row.tracker_closed_at = signal.hit_time
                logger.info('[tracker] Обновлён signal_feature_logs id=%s status=%s', row.id, signal.state.value)
        except Exception as exc:
            logger.error('[tracker] Ошибка записи tracker_* в БД: %s', exc)

    @staticmethod
    async def _maybe_await(callback: AncillaryCallback, *args: Any) -> None:
        result = callback(*args)
        if hasattr(result, '__await__'):
            await result

    def build_entry_event_payload(self, signal: TrackedSignal) -> dict[str, Any]:
        return {
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'source_ai': signal.source_ai,
            'fill_price': signal.fill_price,
            'entry_price': signal.entry_price if signal.entry_price != 'market' else signal.fill_price,
            'leverage': signal.leverage,
            'tp': signal.tp_price,
            'sl': signal.sl_price,
            'entry_time': (signal.actual_entry_time or dt.datetime.now(tz=dt.timezone.utc)).isoformat().replace('+00:00', 'Z'),
            'signal_time': signal.signal_time.isoformat().replace('+00:00', 'Z'),
            'asset_class': signal.asset_class,
        }

    def build_outcome_payload(
        self,
        signal: TrackedSignal,
        pnl_usdt: float | None,
        pnl_percent: float | None,
    ) -> dict[str, Any]:
        return {
            'signal_id': signal.signal_id,
            'source_ai': signal.source_ai,
            'symbol': signal.symbol,
            'asset_class': signal.asset_class,
            'status': signal.state.value,
            'direction': signal.direction,
            'entry_price': signal.fill_price,
            'hit_price': signal.hit_price,
            'tp': signal.tp_price,
            'sl': signal.sl_price,
            'leverage': signal.leverage,
            'pnl_usdt': pnl_usdt,
            'pnl_percent': pnl_percent,
            'duration_seconds': signal.duration_seconds,
            'closed_at': (signal.hit_time or dt.datetime.now(tz=dt.timezone.utc)).isoformat().replace('+00:00', 'Z'),
        }
