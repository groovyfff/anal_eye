"""Trading-symbol helpers for dynamic pair support."""

from __future__ import annotations

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
