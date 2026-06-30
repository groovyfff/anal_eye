from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from shared.binance_symbols import (
    discover_top_futures_symbols,
    is_excluded_symbol,
    manual_symbols_from_env,
    resolve_binance_symbols,
)


def test_is_excluded_symbol_filters_leveraged_tokens() -> None:
    assert is_excluded_symbol("BTCUPUSDT", "USDT") is True
    assert is_excluded_symbol("ETHDOWNUSDT", "USDT") is True
    assert is_excluded_symbol("BTCUSDT", "USDT") is False
    assert is_excluded_symbol("ETHUSDT", "USDT") is False


def test_manual_symbols_from_env_prefers_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYMBOLS", "btcusdt, ethusdt ,solusdt")
    monkeypatch.setenv("BINANCE_SYMBOLS", "XRPUSDT")
    assert manual_symbols_from_env() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_manual_symbols_from_env_falls_back_to_binance_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYMBOLS", raising=False)
    monkeypatch.setenv("BINANCE_SYMBOLS", "SOLUSDT,BNBUSDT")
    assert manual_symbols_from_env() == ["SOLUSDT", "BNBUSDT"]


def test_discover_top_futures_symbols_limits_and_sorts() -> None:
    exchange_info = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "SOLUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "BTCUPUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "DELISTEDUSDT", "status": "BREAK", "contractType": "PERPETUAL"},
        ]
    }
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "1000"},
        {"symbol": "ETHUSDT", "quoteVolume": "5000"},
        {"symbol": "SOLUSDT", "quoteVolume": "2500"},
        {"symbol": "BTCUPUSDT", "quoteVolume": "99999"},
    ]

    def _fake_fetch(path: str, _rest: str):
        if path == "/fapi/v1/exchangeInfo":
            return exchange_info
        if path == "/fapi/v1/ticker/24hr":
            return tickers
        raise AssertionError(path)

    with patch("shared.binance_symbols._fetch_futures_json", side_effect=_fake_fetch):
        symbols = discover_top_futures_symbols(limit=2, quote_asset="USDT")

    assert symbols == ["ETHUSDT", "SOLUSDT"]


def test_resolve_binance_symbols_manual_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
    assert resolve_binance_symbols() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_resolve_binance_symbols_discovers_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYMBOLS", raising=False)
    monkeypatch.delenv("BINANCE_SYMBOLS", raising=False)
    monkeypatch.setenv("SYMBOL_LIMIT", "200")

    with patch(
        "shared.binance_symbols.discover_top_futures_symbols",
        return_value=["BTCUSDT"] + [f"SYM{i}USDT" for i in range(199)],
    ) as discover:
        symbols = resolve_binance_symbols()

    discover.assert_called_once()
    assert len(symbols) == 200
