from __future__ import annotations

from src.kline_parser import parse_kline_message


def test_parse_combined_stream_kline() -> None:
    message = {
        "stream": "btcusdt@kline_1h",
        "data": {
            "e": "kline",
            "E": 1778795999000,
            "s": "BTCUSDT",
            "k": {
                "t": 1778792400000,
                "T": 1778795999000,
                "o": "81000.00",
                "h": "81500.00",
                "l": "80900.00",
                "c": "81396.61",
                "v": "1234.5",
                "x": False,
            },
        },
    }
    live, raw = parse_kline_message(message, market="futures", timeframe="1h")
    assert live["symbol"] == "BTCUSDT"
    assert live["asset_class"] == "crypto"
    assert live["price"] == 81396.61
    assert live["ts"] == 1778795999000
    assert live["is_candle_closed"] is False
    assert live["raw_stream"] == "btcusdt@kline_1h"
    assert raw is not None
    assert raw["kline"]["close"] == 81396.61


def test_parse_direct_kline_event() -> None:
    message = {
        "e": "kline",
        "E": 1000,
        "s": "ETHUSDT",
        "k": {
            "t": 900,
            "T": 1000,
            "o": "1",
            "h": "2",
            "l": "0.5",
            "c": "1.5",
            "v": "10",
            "x": True,
        },
    }
    live, _ = parse_kline_message(message, market="futures", timeframe="1h", default_stream="ethusdt@kline_1h")
    assert live["symbol"] == "ETHUSDT"
    assert live["is_candle_closed"] is True
