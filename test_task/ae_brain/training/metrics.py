"""Backtest and training evaluation metrics (EV-first, not accuracy-only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss, precision_score


@dataclass
class MetricsBundle:
    net_pnl_usd: float = 0.0
    expected_ev_usd: float = 0.0
    realized_ev_usd: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    trade_count: int = 0
    long_count: int = 0
    short_count: int = 0
    skip_count: int = 0
    brier: float | None = None
    precision_at_conf_70: float | None = None
    per_symbol: dict = field(default_factory=dict)
    ev_by_confidence_bucket: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "net_pnl_usd": self.net_pnl_usd,
            "expected_ev_usd": self.expected_ev_usd,
            "realized_ev_usd": self.realized_ev_usd,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "win_rate": self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "trade_count": self.trade_count,
            "long_count": self.long_count,
            "short_count": self.short_count,
            "skip_count": self.skip_count,
            "brier": self.brier,
            "precision_at_conf_70": self.precision_at_conf_70,
            "per_symbol": self.per_symbol,
            "ev_by_confidence_bucket": self.ev_by_confidence_bucket,
            **self.extra,
        }


def max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.maximum(peak, 1e-9)
    return float(np.max(dd))


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 365 * 24) -> float:
    if returns.size < 2:
        return 0.0
    std = float(returns.std())
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: float = 365 * 24) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0]
    dstd = float(downside.std()) if downside.size else 1e-12
    if dstd < 1e-12:
        return 0.0
    return float(returns.mean() / dstd * np.sqrt(periods_per_year))


def profit_factor(pnls: np.ndarray) -> float:
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return float(gains) if gains > 0 else 0.0
    return float(gains / losses)


def precision_at_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float = 0.70,
) -> float | None:
    mask = confidence >= threshold
    if mask.sum() < 5:
        return None
    return float(precision_score(y_true[mask], y_pred[mask], zero_division=0))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if y_true.size < 5:
        return None
    return float(brier_score_loss(y_true, y_prob))


def ev_by_confidence_bucket(
    expected_values: np.ndarray,
    confidence: np.ndarray,
    *,
    edges: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 1.01),
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence >= lo) & (confidence < hi)
        key = f"{lo:.2f}-{hi:.2f}"
        out[key] = float(expected_values[m].mean()) if m.any() else 0.0
    return out


def summarize_trades(
    decisions: np.ndarray,
    pnls: np.ndarray,
    expected_evs: np.ndarray,
    symbols: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> MetricsBundle:
    """Aggregate trade-level metrics from arrays of decisions and PnLs."""
    m = MetricsBundle()
    actionable = np.isin(decisions, ["LONG", "SHORT"])
    m.trade_count = int(actionable.sum())
    m.long_count = int((decisions == "LONG").sum())
    m.short_count = int((decisions == "SHORT").sum())
    m.skip_count = int((decisions == "SKIP").sum())
    if actionable.any():
        ap = pnls[actionable]
        m.net_pnl_usd = float(ap.sum())
        m.expected_ev_usd = float(expected_evs[actionable].sum())
        m.realized_ev_usd = m.net_pnl_usd
        m.profit_factor = profit_factor(ap)
        m.win_rate = float((ap > 0).mean())
        m.avg_trade_return = float(ap.mean())
        eq = np.cumsum(ap)
        m.max_drawdown = max_drawdown(eq)
        rets = ap / max(np.abs(ap).mean(), 1e-9)
        m.sharpe = sharpe_ratio(rets)
        m.sortino = sortino_ratio(rets)
    if confidence is not None and actionable.any():
        y_true = (pnls[actionable] > 0).astype(int)
        y_pred = (expected_evs[actionable] > 0).astype(int)
        m.precision_at_conf_70 = precision_at_confidence(y_true, y_pred, confidence[actionable])
        m.ev_by_confidence_bucket = ev_by_confidence_bucket(expected_evs, confidence)
    if symbols is not None:
        for sym in sorted(set(symbols)):
            sm = symbols == sym
            m.per_symbol[sym] = {
                "net_pnl_usd": float(pnls[sm].sum()),
                "trades": int(np.isin(decisions[sm], ["LONG", "SHORT"]).sum()),
            }
    return m
