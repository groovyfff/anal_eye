#!/usr/bin/env python3
"""Smoke test for the REST closed-candle fallback poller.

Proves — without waiting an hour or hitting the network — that:
  1. A fresh closed BTCUSDT candle fetched via REST publishes exactly one candidate.
  2. A duplicate poll of the same closed candle does NOT publish again (dedup).
  3. The resulting payload is accepted by the AE Brain normalizer
     (``normalize_candidate(...).skip_reason is None``).

The REST fetcher is faked with synthetic candles; no real Binance call is made.
"""

from __future__ import annotations

import asyncio
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

from ae_brain.messaging.candidate_normalizer import normalize_candidate  # noqa: E402
from src.candle_buffer import Candle, CandleBuffer  # noqa: E402
from src.config import ServiceConfig  # noqa: E402
from src.dedup_store import DedupStore  # noqa: E402
from src.binance_ws import BinanceCandidateStream  # noqa: E402
from src.rest_closed_candle_poller import RestClosedCandlePoller, latest_closed_candle  # noqa: E402

_HISTORICAL = 200  # closed candles in the buffer/window


def _gen_candles(n_closed: int, *, open_at_last_close: bool, last_close_ts: int) -> list[Candle]:
    """Return ``n_closed`` closed candles ending at ``last_close_ts``, plus
    optionally one currently-open candle that must NOT be published."""
    out: list[Candle] = []
    # Closed candles: each is 1h; the last closed candle ends at last_close_ts.
    last_open = last_close_ts - 3_599_999  # close_time of the last closed candle
    start_open = last_open - (n_closed - 1) * 3_600_000
    for i in range(n_closed):
        ot = start_open + i * 3_600_000
        out.append(
            Candle(
                timestamp=ot,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
                closed=True,
                close_time=ot + 3_599_999,
            )
        )
    if open_at_last_close:
        # Currently-forming candle (close_time in the future) — must be ignored.
        ot = last_open + 3_600_000
        out.append(
            Candle(
                timestamp=ot,
                open=200.0,
                high=201.0,
                low=199.0,
                close=200.5,
                volume=500.0,
                closed=False,
                close_time=ot + 3_599_999,
            )
        )
    return out


def main() -> int:
    # Single-symbol scope keeps the smoke deterministic (one fetcher, one publish).
    os.environ.setdefault("ANAL_EYES_ALLOWED_SYMBOLS", "BTCUSDT")
    os.environ.setdefault("SYMBOLS", "BTCUSDT")
    config = ServiceConfig.from_env(symbols=["BTCUSDT"])
    now_ms = 1_700_800_000_000  # fixed "now" so the test is deterministic
    buffer = CandleBuffer(max_candles=config.max_candles)

    published: list[dict] = []

    class _FakePublisher:
        async def publish_candidate(self, payload: dict) -> None:
            published.append(payload)

    # Track poll iteration so the fake fetcher can serve a *new* closed candle
    # on the first poll (simulating an hour closing) and the same data on the
    # second poll (dedup case).
    state = {"candles": _gen_candles(_HISTORICAL, open_at_last_close=True, last_close_ts=now_ms)}

    def _fake_fetch(**kwargs):
        return list(state["candles"])

    async def _run() -> int:
        with tempfile.TemporaryDirectory() as tmp:
            dedup = DedupStore(Path(tmp) / "dedup.db")
            stream = BinanceCandidateStream(config, buffer, _FakePublisher(), dedup)
            poller = RestClosedCandlePoller(
                config,
                buffer,
                stream,
                ws_health_getter=lambda: False,  # simulate dead WS -> fallback active
                fetcher=_fake_fetch,
                sleep_fn=lambda *_a, **_k: asyncio.sleep(0),  # no real waiting
            )

            print("=== poll 1: fresh closed candle -> expect 1 publish ===")
            await poller._poll_once()
            print(f"published_count={len(published)}")
            assert len(published) == 1, f"expected 1 publish, got {len(published)}"

            print("=== poll 2: same closed candle -> expect dedup skip, 0 new publishes ===")
            await poller._poll_once()
            print(f"published_count={len(published)}")
            assert len(published) == 1, f"expected still 1 publish (dedup), got {len(published)}"

            # Verify the latest-closed helper correctly excludes the open candle.
            latest = latest_closed_candle(state["candles"], now_ms=now_ms)
            assert latest is not None
            assert latest.close_time < now_ms, "open candle must not be selected as latest closed"

            dedup.close()

        print("=== AE Brain normalizer accepts the fallback payload ===")
        norm = normalize_candidate(published[0], min_composite_score=0.0)
        print(f"skip_reason={norm.skip_reason!r} symbol={norm.payload.get('symbol') if norm.payload else None}")
        assert norm.skip_reason is None, f"normalizer rejected: {norm.skip_reason}"
        assert norm.payload is not None
        assert norm.payload["symbol"] == "BTCUSDT"
        assert norm.payload["interval"] == "1h"
        assert len(norm.payload["candles"]) == _HISTORICAL

        print("\nALL REST-FALLBACK SMOKE CHECKS PASSED")
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
