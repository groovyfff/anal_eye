"""Trading-symbol helpers for dynamic pair support."""

from __future__ import annotations

from pathlib import Path

# Longest suffixes first so e.g. USDT wins over USD.
_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "FDUSD",
    "BUSD",
    "TUSD",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
    "USD",
)

UNKNOWN_SYMBOL = "UNKNOWN"

_UNIVERSE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "symbols_universe.yaml"

# Evidenced in commit 6d606390 test_task/tests/test_dynamic_symbols.py
_FALLBACK_UNIVERSE: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
)


def load_default_universe() -> tuple[str, ...]:
    """Load the default symbol universe from config/symbols_universe.yaml."""
    if not _UNIVERSE_CONFIG.exists():
        return _FALLBACK_UNIVERSE
    symbols: list[str] = []
    for line in _UNIVERSE_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            sym = stripped[2:].strip().upper()
            if sym:
                symbols.append(sym)
    return tuple(symbols) if symbols else _FALLBACK_UNIVERSE


DEFAULT_SYMBOL_UNIVERSE: tuple[str, ...] = load_default_universe()


def default_allowed_symbols_csv() -> str:
    return ",".join(DEFAULT_SYMBOL_UNIVERSE)


def parse_symbol_list(raw: str | None) -> list[str]:
    """Parse comma-separated symbols; empty -> default universe."""
    if not raw or not str(raw).strip():
        return list(DEFAULT_SYMBOL_UNIVERSE)
    return [part.strip().upper() for part in str(raw).split(",") if part.strip()]


def normalize_symbol(symbol: str | None) -> str:
    """Uppercase and strip a trading symbol; missing input becomes ``UNKNOWN``."""
    if symbol is None:
        return UNKNOWN_SYMBOL
    cleaned = str(symbol).strip().upper()
    return cleaned or UNKNOWN_SYMBOL


def extract_base_asset(symbol: str | None) -> str:
    """Derive the base asset from a pair (``ETHUSDT`` -> ``ETH``).

    Traditional tickers such as ``AAPL`` or ``GC=F`` are returned unchanged when
    no known quote suffix matches.
    """
    sym = normalize_symbol(symbol)
    if sym == UNKNOWN_SYMBOL:
        return UNKNOWN_SYMBOL
    for quote in _QUOTE_SUFFIXES:
        if sym.endswith(quote) and len(sym) > len(quote):
            return sym[: -len(quote)]
    return sym


def require_symbol(symbol: str | None) -> str:
    """Validate inbound symbol; raise ``ValueError`` when absent."""
    cleaned = normalize_symbol(symbol)
    if cleaned == UNKNOWN_SYMBOL:
        raise ValueError("missing_symbol: candidate payload must include a non-empty 'symbol' field")
    return cleaned
