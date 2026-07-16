"""Tests for shared production symbol universe."""

from __future__ import annotations

import pytest

from shared.symbol_universe import (
    DEFAULT_PRODUCTION_UNIVERSE,
    default_allowed_symbols,
    is_symbol_allowed,
    resolve_production_symbols,
)


def test_default_universe_is_six_symbols() -> None:
    assert default_allowed_symbols() == frozenset(DEFAULT_PRODUCTION_UNIVERSE)
    assert len(DEFAULT_PRODUCTION_UNIVERSE) == 6


@pytest.mark.parametrize("symbol", ["ADAUSDT", "AVAXUSDT", "LINKUSDT"])
def test_out_of_universe_symbols_rejected(symbol: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANAL_EYES_ALLOWED_SYMBOLS", raising=False)
    monkeypatch.delenv("ALLOWED_SYMBOLS", raising=False)
    monkeypatch.delenv("AEB_ALLOWED_SYMBOLS", raising=False)
    assert not is_symbol_allowed(symbol)


def test_manual_symbols_filtered_by_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ANAL_EYES_ALLOWED_SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT",
    )
    manual = ["BTCUSDT", "ADAUSDT", "ETHUSDT", "LINKUSDT"]
    filtered = resolve_production_symbols(manual)
    assert filtered == ["BTCUSDT", "ETHUSDT"]
    assert "ADAUSDT" not in filtered
    assert "LINKUSDT" not in filtered
