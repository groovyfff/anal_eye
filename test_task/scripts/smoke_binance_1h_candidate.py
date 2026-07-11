#!/usr/bin/env python3
"""Smoke test for 1h closed-candle candidate pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINANCE_SRC = ROOT / "backend" / "Annnneqwe" / "analeyes" / "services" / "binance-candidate-service"
SHARED_SRC = ROOT / "backend" / "Annnneqwe" / "analeyes" / "shared" / "src"
for p in (str(BINANCE_SRC), str(SHARED_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.binance_ws import BinanceCandidateStream
from src.candle_buffer import Candle, CandleBuffer
from src.config import ServiceConfig
from src.dedup_store import DedupStore

_GATE_PATH = (
    ROOT
    / "backend"
    / "Annnneqwe"
    / "analeyes"
    / "services"
    / "notification-service"
    / "src"
    / "logic"
    / "telegram"
    / "telegram_gate.py"
)
_spec = importlib.util.spec_from_file_location("telegram_gate", _GATE_PATH)
assert _spec and _spec.loader
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)
evaluate_telegram_signal = _gate.evaluate_telegram_signal


def _candles(n: int) -> list[Candle]:
    start = 1_700_000_000_000
    return [
        Candle(
            timestamp=start + i * 3_600_000,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
            closed=True,
            close_time=start + i * 3_600_000 + 3_599_999,
        )
        for i in range(n)
    ]


def _kline(symbol: str, open_time: int, closed: bool) -> str:
    return json.dumps(
        {
            "e": "kline",
            "E": open_time + 3_599_999,
            "s": symbol,
            "k": {
                "t": open_time,
                "T": open_time + 3_599_999,
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100.5",
                "v": "1000",
                "x": closed,
            },
        }
    )


async def run_smoke() -> int:
    os.environ.setdefault(
        "ANAL_EYES_ALLOWED_SYMBOLS",
        "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT",
    )
    config = ServiceConfig.from_env(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    )
    buffer = CandleBuffer(max_candles=200)
    buffer.load_bootstrap("BTCUSDT", _candles(200))

    published: list[dict] = []

    class _FakePublisher:
        async def publish_candidate(self, payload: dict) -> None:
            published.append(payload)

    with tempfile.TemporaryDirectory() as tmp:
        dedup = DedupStore(Path(tmp) / "dedup.db")
        stream = BinanceCandidateStream(config, buffer, _FakePublisher(), dedup)
        open_time = 1_700_720_000_000

        print("=== closed BTCUSDT 1h -> expect publish ===")
        await stream._handle_message(_kline("BTCUSDT", open_time, True))
        print(f"published_count={len(published)}")

        print("=== duplicate BTCUSDT candle -> expect dedup skip ===")
        await stream._handle_message(_kline("BTCUSDT", open_time, True))
        print(f"published_count={len(published)}")

        print("=== non-closed BTCUSDT -> expect skip ===")
        await stream._handle_message(_kline("BTCUSDT", open_time + 3_600_000, False))
        print(f"published_count={len(published)}")

        print("=== ADAUSDT closed -> expect reject ===")
        buffer.load_bootstrap("ADAUSDT", _candles(200))
        await stream._handle_message(_kline("ADAUSDT", open_time, True))
        print(f"published_count={len(published)}")

        dedup.close()

    allowed = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"})
    ok_low, reason_low = evaluate_telegram_signal(
        {"symbol": "BTCUSDT", "decision": "LONG", "confidence": 0.39},
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    print(f"=== notification low confidence -> rejected={not ok_low} reason={reason_low} ===")

    ok_skip, reason_skip = evaluate_telegram_signal(
        {"symbol": "BTCUSDT", "decision": "SKIP", "confidence": 0.95},
        allowed_symbols=allowed,
        min_confidence=0.70,
    )
    print(f"=== notification SKIP -> rejected={not ok_skip} reason={reason_skip} ===")

    return 0 if len(published) == 1 and not ok_low and not ok_skip else 1


def main() -> int:
    return asyncio.run(run_smoke())


if __name__ == "__main__":
    sys.exit(main())
