"""Synthetic market data generator.

Produces realistic-ish OHLCV + microstructure candles via a regime-switching
geometric Brownian motion with stochastic volatility and momentum bursts. Used
to smoke-test the full train -> serve pipeline without a live data feed.

NOT for production signal generation - only for plumbing/CI validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_candles(
    n: int = 20_000,
    start_price: float = 30_000.0,
    seed: int = 7,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Regime-switching drift + stochastic vol.
    regimes = rng.integers(0, 3, size=n // 200 + 1)  # 0 flat, 1 bull, 2 bear
    drift_map = {0: 0.0, 1: 0.00008, 2: -0.00008}
    drift = np.repeat([drift_map[r] for r in regimes], 200)[:n]

    vol = 0.004 * np.exp(0.3 * np.cumsum(rng.normal(0, 0.02, n)))
    vol = np.clip(vol, 0.0015, 0.03)
    shocks = rng.normal(0, 1, n) * vol + drift
    # momentum bursts
    burst = (rng.random(n) < 0.01) * rng.normal(0, 0.02, n)
    log_ret = shocks + burst

    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol)))
    volume = np.abs(rng.normal(1000, 300, n)) * (1 + 5 * np.abs(log_ret))
    taker_buy = volume * np.clip(0.5 + 2 * log_ret, 0.05, 0.95)

    open_interest = np.abs(np.cumsum(rng.normal(0, 50, n)) + 1e6)
    funding = np.clip(rng.normal(0.0001, 0.0001, n), -0.003, 0.003)
    basis = rng.normal(0, 0.0005, n)
    bid_size = np.abs(rng.normal(50, 15, n))
    ask_size = np.abs(rng.normal(50, 15, n))
    spread = np.abs(rng.normal(close * 1e-5, close * 5e-6))
    long_liq = np.where(rng.random(n) < 0.02, np.abs(rng.normal(1e5, 5e4, n)), 0.0)
    short_liq = np.where(rng.random(n) < 0.02, np.abs(rng.normal(1e5, 5e4, n)), 0.0)
    trade_count = np.abs(rng.normal(500, 150, n))

    ts = pd.date_range("2023-01-01", periods=n, freq=f"{interval_minutes}min", tz="UTC")

    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_buy_volume": taker_buy,
            "open_interest": open_interest,
            "funding_rate": funding,
            "basis": basis,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread": spread,
            "long_liq_notional": long_liq,
            "short_liq_notional": short_liq,
            "trade_count": trade_count,
        }
    )
