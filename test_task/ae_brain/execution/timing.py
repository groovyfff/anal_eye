"""Execution timing — separate signal candle T from fill price/time.

Rules
-----
* Features and model inference may use data through closed candle T (inclusive).
* Simulated or live entry uses execution_price at T+1 open (backtest) or mark
  after close with slippage (live), never the unavailable future close of T+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ae_brain.contracts import Side


def _to_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ms = int(value)
        if ms < 10_000_000_000:
            return ms
        return ms
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        normalized = text.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def ms_to_iso_utc(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _last_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ExecutionTiming:
    signal_candle_open_time: str
    signal_candle_close_time: str
    execution_time: str
    execution_price_source: str
    execution_price: float
    signal_reference_price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_candle_open_time": self.signal_candle_open_time,
            "signal_candle_close_time": self.signal_candle_close_time,
            "execution_time": self.execution_time,
            "execution_price_source": self.execution_price_source,
            "execution_price": self.execution_price,
            "signal_reference_price": self.signal_reference_price,
        }


def _apply_slippage(price: float, *, side: Side | None, slippage_bps: float) -> float:
    slip = max(0.0, slippage_bps) / 10_000.0
    if side == Side.LONG:
        return price * (1.0 + slip)
    if side == Side.SHORT:
        return price * (1.0 - slip)
    return price * (1.0 + slip)


def candles_for_features(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candle window used for features — through closed signal candle T only."""
    return list(candles)


def resolve_execution_timing(
    candles: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    *,
    slippage_bps: float = 5.0,
    side: Side | None = None,
) -> ExecutionTiming:
    """Resolve execution fill from closed signal candle T and optional T+1 open."""
    if not candles:
        raise ValueError("missing_candles_for_execution_timing")

    meta = meta or {}
    signal_row = dict(candles[-1])
    signal_reference_price = _last_float(
        signal_row,
        "close",
        _last_float(meta, "current_price", 0.0),
    )

    open_ms = _to_ms(
        meta.get("signal_candle_open_time")
        or meta.get("candle_open_time")
        or signal_row.get("timestamp")
        or signal_row.get("open_time")
        or signal_row.get("ts")
    )
    close_ms = _to_ms(
        meta.get("signal_candle_close_time")
        or meta.get("candle_close_time")
        or signal_row.get("close_time")
        or signal_row.get("event_time")
    )
    if open_ms is None:
        open_ms = _to_ms(signal_row.get("timestamp") or signal_row.get("ts")) or 0
    if close_ms is None:
        close_ms = open_ms + 3_599_999

    next_open = meta.get("next_candle_open")
    next_open_ms = _to_ms(meta.get("next_candle_open_time"))
    explicit_source = str(meta.get("execution_price_source") or "").strip().lower()
    explicit_price = meta.get("execution_price")

    if next_open is not None:
        execution_price = float(next_open)
        execution_time = ms_to_iso_utc(next_open_ms) if next_open_ms else ms_to_iso_utc(close_ms + 1)
        source = "next_open"
    elif explicit_price is not None and explicit_source:
        execution_price = float(explicit_price)
        execution_time = str(meta.get("execution_time") or ms_to_iso_utc(close_ms + 1))
        source = explicit_source
    else:
        mark = _last_float(meta, "mark_price", signal_reference_price)
        if mark <= 0:
            mark = signal_reference_price
        execution_price = _apply_slippage(mark, side=side, slippage_bps=slippage_bps)
        execution_time = str(meta.get("execution_time") or ms_to_iso_utc(close_ms + 1))
        source = "live_mark_after_close"

    if execution_price <= 0:
        raise ValueError("invalid_execution_price")

    return ExecutionTiming(
        signal_candle_open_time=ms_to_iso_utc(open_ms),
        signal_candle_close_time=ms_to_iso_utc(close_ms),
        execution_time=execution_time,
        execution_price_source=source,
        execution_price=execution_price,
        signal_reference_price=signal_reference_price,
    )


def reprice_tp_sl_for_entry(
    *,
    side: Side,
    old_entry: float,
    new_entry: float,
    take_profit: float,
    stop_loss: float,
) -> tuple[float, float]:
    """Shift TP/SL by entry delta so distances are preserved (no lookahead re-fit)."""
    if old_entry <= 0:
        return take_profit, stop_loss
    delta = new_entry - old_entry
    return take_profit + delta, stop_loss + delta


def compare_entry_pnl_delta(
    *,
    side: Side,
    signal_close_entry: float,
    execution_entry: float,
    take_profit: float,
    stop_loss: float,
    notional_usd: float,
) -> float:
    """USD PnL difference between filling at signal close vs execution entry (zero at TP/SL hit)."""
    if signal_close_entry <= 0 or execution_entry <= 0 or notional_usd <= 0:
        return 0.0
    qty_signal = notional_usd / signal_close_entry
    qty_exec = notional_usd / execution_entry
    if side == Side.LONG:
        return (execution_entry - signal_close_entry) * qty_exec
    return (signal_close_entry - execution_entry) * qty_exec
