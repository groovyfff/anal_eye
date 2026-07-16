"""Production crypto symbol allowlist shared across AnalEyes services."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTION_UNIVERSE: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
)

_ENV_KEYS: tuple[str, ...] = (
    "ANAL_EYES_ALLOWED_SYMBOLS",
    "ALLOWED_SYMBOLS",
    "AEB_ALLOWED_SYMBOLS",
)


def parse_symbol_csv(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    return [part.strip().upper() for part in str(raw).split(",") if part.strip()]


def allowed_symbols_from_env() -> frozenset[str] | None:
    """Return explicit allowlist from env, or ``None`` when unset."""
    for key in _ENV_KEYS:
        raw = os.environ.get(key)
        symbols = parse_symbol_csv(raw)
        if symbols:
            return frozenset(symbols)
    return None


def default_allowed_symbols() -> frozenset[str]:
    return frozenset(allowed_symbols_from_env() or DEFAULT_PRODUCTION_UNIVERSE)


def allowed_symbols_csv() -> str:
    return ",".join(sorted(default_allowed_symbols()))


def is_symbol_allowed(symbol: str | None, allowed: frozenset[str] | None = None) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    universe = allowed if allowed is not None else default_allowed_symbols()
    return sym in universe


def filter_symbols(symbols: list[str], allowed: frozenset[str] | None = None) -> list[str]:
    universe = allowed if allowed is not None else default_allowed_symbols()
    return [sym for sym in symbols if sym.upper() in universe]


def resolve_production_symbols(manual: list[str] | None = None) -> list[str]:
    """Resolve the effective symbol list for candidate producers.

    Priority:
    1. ``SYMBOLS`` / ``BINANCE_SYMBOLS`` manual list, intersected with allowlist
    2. ``ANAL_EYES_ALLOWED_SYMBOLS`` / ``ALLOWED_SYMBOLS`` / ``AEB_ALLOWED_SYMBOLS``
    3. Default 6-symbol production universe
    """
    universe = default_allowed_symbols()
    if manual:
        filtered = filter_symbols(manual, universe)
        if len(filtered) < len(manual):
            rejected = [sym for sym in manual if sym.upper() not in universe]
            logger.info(
                "symbol_universe_filtered manual_count=%s allowed_count=%s rejected=%s",
                len(manual),
                len(filtered),
                ",".join(rejected[:10]),
            )
        return filtered
    return sorted(universe)
