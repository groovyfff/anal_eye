"""Isolated boundary probe for ``external-markets-service``.

This module is executed as a **subprocess** by ``test_l3_e2e_flow.py`` rather
than imported directly. The reason is namespace isolation: both
``tracker-service`` and ``external-markets-service`` expose a top-level ``src``
package, so they cannot coexist on ``sys.path`` inside a single interpreter.
Running the external-markets check in its own process with a dedicated
``PYTHONPATH`` (``shared/src:services/external-markets-service``) keeps the
``src`` import unambiguous.

The probe constructs the *real* :class:`ExternalMarketsService` (offline, no DB,
no broker connect) and asserts the session-boundary invariant:

    Scanning a traditional asset while its market is **closed** must suppress
    candidate generation entirely - no OHLCV fetch, no publish.

On success it prints ``BOUNDARY_OK`` and exits 0; any failure raises / exits
non-zero so the parent test surfaces it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys


def _build_settings() -> dict:
    return {
        "rabbitmq": {
            "url": "amqp://guest:guest@127.0.0.1:5672/",
            "exchange": "analeyes_exchange",
            "connect_retries": 1,
            "connect_retry_delay_s": 0,
        },
        "data_provider": {"name": "yahoo_finance"},
        "market_hours": {
            "timezone": "America/New_York",
            "use_fixed_utc_window": True,
            "stock_open_utc": "14:30",
            "stock_close_utc": "21:00",
            "pre_market_enabled": False,
            "after_hours_enabled": False,
            "metal_breaks_utc": [],
        },
        "watchlist": {"stocks": [{"symbol": "AAPL", "name": "Apple Inc."}]},
        "database": {"enabled": False},
        "logging": {"service_name": "external-markets-service", "level": "ERROR"},
        "main_timeframe": "5m",
        "history_depth": 64,
        "scan_workers": 1,
    }


async def _run() -> None:
    from src.main import ExternalMarketsService  # external-markets `src`

    service = ExternalMarketsService(_build_settings())

    item = next(i for i in service.scan_candidates if i["symbol"] == "AAPL")
    assert item["asset_class"] == "stock"

    # Saturday 2026-06-27 12:00 UTC -> NY Saturday -> stock market closed.
    closed_now = dt.datetime(2026, 6, 27, 12, 0, tzinfo=dt.timezone.utc)
    assert service.market_hours.is_market_open("stock", now_utc=closed_now) is False, (
        "precondition failed: market should be closed on Saturday"
    )

    fetched: list[str] = []
    published: list[dict] = []

    async def _fetch_spy(*args, **kwargs):  # would hit the network if reached
        fetched.append(kwargs.get("symbol") or (args[0] if args else "?"))
        raise AssertionError("OHLCV fetch must not run while market is closed")

    async def _publish_spy(*args, **kwargs):
        published.append(kwargs or {"args": args})
        return True

    service._fetch_ohlcv = _fetch_spy  # type: ignore[assignment]
    service._publish_json = _publish_spy  # type: ignore[assignment]

    # The market-closed guard sits at the very top of _scan_symbol.
    await service._scan_symbol(item=item, now_utc=closed_now, benchmark_close=None, dxy_close=None)

    assert not fetched, f"market closed but OHLCV was fetched: {fetched}"
    assert not published, f"market closed but a candidate was published: {published}"

    # Positive control: the same symbol IS open inside the fixed UTC window, so
    # the guard would NOT short-circuit (it would proceed to fetch and raise).
    open_now = dt.datetime(2026, 6, 29, 15, 0, tzinfo=dt.timezone.utc)  # Monday
    assert service.market_hours.is_market_open("stock", now_utc=open_now) is True
    await service._scan_symbol(item=item, now_utc=open_now, benchmark_close=None, dxy_close=None)
    assert fetched, "market open but the scan never attempted an OHLCV fetch"

    print("BOUNDARY_OK")


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface failure to the parent test
        print(f"BOUNDARY_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
