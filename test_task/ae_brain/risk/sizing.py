"""Dynamic position sizing and stop placement.

Two independent, composable mechanisms (per spec - NO hardcoded -5% stops):

1. **ATR-based stops** - take-profit / stop-loss distances are multiples of the
   current ATR, so they breathe with realized volatility.
2. **Fractional Kelly sizing** - bet size is a fraction of the Kelly-optimal
   stake, additionally clamped by an ATR volatility-target and hard caps.

A **correlation limit** rejects/scales size when the new position would push
summed correlated exposure beyond the configured budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from ae_brain.config import RiskConfig
from ae_brain.contracts import Side
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.sizing")


@dataclass(slots=True)
class SizingResult:
    position_size_pct: float  # fraction of equity actually allocated (margin)
    notional_usd: float
    leverage: float
    take_profit: float
    stop_loss: float
    stop_distance: float
    kelly_fraction_raw: float
    correlation_scale: float
    rejected_reason: str | None = None


class PositionSizer:
    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    # -- stops ------------------------------------------------------------
    def atr_stops(self, entry: float, atr: float, side: Side) -> tuple[float, float, float]:
        """Return (take_profit, stop_loss, stop_distance) from ATR multiples."""
        stop_dist = max(atr * self._cfg.atr_sl_mult, 1e-9)
        tp_dist = atr * self._cfg.atr_tp_mult
        if side == Side.LONG:
            return entry + tp_dist, entry - stop_dist, stop_dist
        return entry - tp_dist, entry + stop_dist, stop_dist

    # -- Kelly ------------------------------------------------------------
    @staticmethod
    def kelly_fraction(prob_win: float, reward_risk_ratio: float) -> float:
        """Full-Kelly fraction f* = p - (1-p)/b for payoff odds b.

        Returns 0 when the edge is non-positive (never size into negative EV).
        """
        b = max(reward_risk_ratio, 1e-9)
        p = min(max(prob_win, 0.0), 1.0)
        f = p - (1.0 - p) / b
        return max(f, 0.0)

    def volatility_target_cap(self, atr_pct: float) -> float:
        """Cap position fraction so that a 1-ATR move risks a bounded % of equity.

        Higher volatility (atr_pct) -> smaller allowed fraction. Targets ~1%
        equity risk per ATR by default, bounded by configured caps.
        """
        target_risk_per_atr = 0.01
        if atr_pct <= 0:
            return self._cfg.max_position_pct
        return min(self._cfg.max_position_pct, target_risk_per_atr / atr_pct)

    def size(
        self,
        *,
        entry: float,
        atr: float,
        side: Side,
        prob_tp: float,
        reward_risk_ratio: float,
        correlated_exposure: float = 0.0,
    ) -> SizingResult:
        """Compute the full sizing decision."""
        tp, sl, stop_dist = self.atr_stops(entry, atr, side)
        atr_pct = atr / entry if entry > 0 else 0.0

        # Fractional Kelly, then volatility-target + hard caps.
        f_full = self.kelly_fraction(prob_tp, reward_risk_ratio)
        log.info(
            "sizing.kelly",
            prob_tp=round(prob_tp, 6),
            reward_risk_ratio=round(reward_risk_ratio, 4),
            kelly_fraction_raw=round(f_full, 6),
        )
        f_frac = f_full * self._cfg.kelly_fraction
        f_capped = min(f_frac, self.volatility_target_cap(atr_pct), self._cfg.max_position_pct)

        # Correlation limit: scale down as we approach the exposure budget.
        budget = self._cfg.max_correlated_exposure
        rejected = None
        corr_scale = 1.0
        if correlated_exposure >= budget:
            corr_scale = 0.0
            rejected = "correlation_budget_exhausted"
        elif correlated_exposure > 0:
            corr_scale = max(0.0, 1.0 - correlated_exposure / budget)

        position_pct = f_capped * corr_scale
        if position_pct > 0 and position_pct < self._cfg.min_position_pct:
            position_pct = 0.0
            rejected = rejected or "below_min_position"

        # Leverage: use just enough leverage to express the notional implied by
        # the volatility-scaled fraction, capped by max_leverage.
        equity = self._cfg.account_equity_usd
        margin_usd = position_pct * equity
        # Notional sized so stop-distance loss ~ margin (risk parity at stop).
        if stop_dist > 0 and entry > 0:
            risk_implied_notional = margin_usd / (stop_dist / entry)
        else:
            risk_implied_notional = margin_usd
        max_notional = margin_usd * self._cfg.max_leverage
        notional = min(risk_implied_notional, max_notional)
        leverage = (notional / margin_usd) if margin_usd > 0 else 0.0
        leverage = min(max(leverage, 0.0), self._cfg.max_leverage)

        return SizingResult(
            position_size_pct=round(position_pct, 6),
            notional_usd=round(notional, 2),
            leverage=round(leverage, 3),
            take_profit=tp,
            stop_loss=sl,
            stop_distance=stop_dist,
            kelly_fraction_raw=round(f_full, 6),
            correlation_scale=round(corr_scale, 4),
            rejected_reason=rejected,
        )
