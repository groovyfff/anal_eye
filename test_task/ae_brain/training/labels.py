"""EV-aware trading labels and distribution reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ae_brain.config import CostConfig, RiskConfig
from ae_brain.contracts import Side
from ae_brain.risk.costs import CostModel
from ae_brain.layers.meta import CLASS_LONG, CLASS_SHORT
from ae_brain.training.dataset import directional_barrier_labels, relative_vol_scale

LABEL_LONG, LABEL_SHORT, LABEL_SKIP = CLASS_LONG, CLASS_SHORT, 1


@dataclass(frozen=True, slots=True)
class LabelConfig:
    tp_mult: float = 2.5
    sl_mult: float = 1.5
    horizon: int = 24
    min_net_reward_usd: float = 1.0
    account_equity_usd: float = 100_000.0
    holding_hours: float = 8.0


def _net_reward_at_barrier(
    entry: float,
    target: float,
    stop: float,
    side: Side,
    costs: CostModel,
    equity: float,
    funding_rate: float,
) -> float:
    """Approximate net USD reward if TP is hit (gross - costs on a risk-parity notional)."""
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0:
        return -1.0
    margin = 0.01 * equity
    notional = margin / (stop_dist / entry)
    qty = notional / entry
    gross = abs(target - entry) * qty
    trade_costs = costs.estimate(notional, side, funding_rate_8h=funding_rate, holding_hours=8.0)
    return gross - trade_costs.total


def ev_aware_directional_labels(
    candles: pd.DataFrame,
    atr: np.ndarray,
    *,
    cfg: LabelConfig | None = None,
    vol_scale: np.ndarray | None = None,
    funding: np.ndarray | None = None,
) -> np.ndarray:
    """Symmetric LONG/SHORT/SKIP labels gated by positive net reward after costs.

    Starts from first-passage directional barriers, then:
    * LONG kept only if net reward at long TP > min_net_reward_usd
    * SHORT kept only if net reward at short TP > min_net_reward_usd
    * Otherwise SKIP (ambiguous or unprofitable after fees/funding/slippage)
    """
    cfg = cfg or LabelConfig()
    base = directional_barrier_labels(
        candles,
        atr,
        tp_mult=cfg.tp_mult,
        sl_mult=cfg.sl_mult,
        horizon=cfg.horizon,
        vol_scale=vol_scale,
    )
    close = candles["close"].to_numpy(float)
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    fund = funding if funding is not None else np.zeros(len(close))
    costs = CostModel(CostConfig())
    n = len(close)
    out = np.ones(n, dtype=np.int64)
    for i in range(n - 1):
        entry = close[i]
        vs = float(vol_scale[i]) if vol_scale is not None else 1.0
        u_tp = entry + atr[i] * cfg.tp_mult * vs
        d_sl = entry - atr[i] * cfg.sl_mult * vs
        d_tp = entry - atr[i] * cfg.tp_mult * vs
        u_sl = entry + atr[i] * cfg.sl_mult * vs
        fr = float(fund[i])
        long_ev = _net_reward_at_barrier(entry, u_tp, d_sl, Side.LONG, costs, cfg.account_equity_usd, fr)
        short_ev = _net_reward_at_barrier(entry, d_tp, u_sl, Side.SHORT, costs, cfg.account_equity_usd, fr)
        if base[i] == LABEL_LONG and long_ev >= cfg.min_net_reward_usd:
            out[i] = LABEL_LONG
        elif base[i] == LABEL_SHORT and short_ev >= cfg.min_net_reward_usd:
            out[i] = LABEL_SHORT
        else:
            out[i] = LABEL_SKIP
    return out


def label_distribution_report(
    labels: np.ndarray,
    timestamps: pd.Series,
    symbols: np.ndarray,
) -> dict:
    """Aggregate label stats overall, per symbol, and per month."""
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    ts_series = ts if isinstance(ts, pd.Series) else pd.Series(ts)
    months = ts_series.dt.to_period("M").astype(str).to_numpy()
    sym = np.asarray(symbols)
    overall = {
        "LONG": int(np.sum(labels == LABEL_LONG)),
        "SHORT": int(np.sum(labels == LABEL_SHORT)),
        "SKIP": int(np.sum(labels == LABEL_SKIP)),
        "n": int(len(labels)),
    }
    by_symbol: dict[str, dict[str, int]] = {}
    for s in sorted(set(sym)):
        m = sym == s
        by_symbol[s] = {
            "LONG": int(np.sum((labels == LABEL_LONG) & m)),
            "SHORT": int(np.sum((labels == LABEL_SHORT) & m)),
            "SKIP": int(np.sum((labels == LABEL_SKIP) & m)),
        }
    by_time: dict[str, dict[str, int]] = {}
    for period in sorted(set(months)):
        m = months == period
        by_time[period] = {
            "LONG": int(np.sum((labels == LABEL_LONG) & m)),
            "SHORT": int(np.sum((labels == LABEL_SHORT) & m)),
            "SKIP": int(np.sum((labels == LABEL_SKIP) & m)),
        }
    long_n = overall["LONG"]
    short_n = overall["SHORT"]
    return {
        "label_distribution_overall": overall,
        "label_distribution_by_symbol": by_symbol,
        "label_distribution_by_time": by_time,
        "long_short_balance": {
            "long_pct": long_n / max(overall["n"], 1),
            "short_pct": short_n / max(overall["n"], 1),
            "skip_ratio": overall["SKIP"] / max(overall["n"], 1),
        },
    }


def compute_labels_for_frame(
    candles: pd.DataFrame,
    feats_atr: np.ndarray,
    *,
    cfg: LabelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (directional_labels, vol_scale)."""
    atr = np.where(feats_atr <= 0, candles["close"].to_numpy(float) * 0.005, feats_atr)
    vol_scale = relative_vol_scale(
        pd.Series(atr / candles["close"].to_numpy(float)).fillna(0.01).to_numpy()
    )
    funding = candles["funding_rate"].to_numpy(float) if "funding_rate" in candles else None
    labels = ev_aware_directional_labels(candles, atr, cfg=cfg, vol_scale=vol_scale, funding=funding)
    return labels, vol_scale


def side_specific_profitable_labels(
    candles: pd.DataFrame,
    atr: np.ndarray,
    *,
    long_cfg: LabelConfig,
    short_cfg: LabelConfig,
    vol_scale: np.ndarray | None = None,
    funding: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent LONG/SHORT profitable binary labels (SKIP bars -> 0 for both sides)."""
    long_labels = ev_aware_directional_labels(
        candles, atr, cfg=long_cfg, vol_scale=vol_scale, funding=funding
    )
    short_labels = ev_aware_directional_labels(
        candles, atr, cfg=short_cfg, vol_scale=vol_scale, funding=funding
    )
    n = len(candles)
    y_long = np.zeros(n, dtype=np.int64)
    y_short = np.zeros(n, dtype=np.int64)
    y_long[long_labels == LABEL_LONG] = 1
    y_short[short_labels == LABEL_SHORT] = 1
    return y_long, y_short
