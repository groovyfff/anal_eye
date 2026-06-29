"""Expected-Value gate - the single most important guardrail in the system.

No trade may be published unless it passes this gate. The gate enforces the
*exact* formula mandated by the system spec:

    expected_value = (prob_tp * net_reward) - (prob_sl * net_risk)
    is_positive_ev = expected_value > 0

Where:
  * ``prob_tp`` - calibrated probability of hitting take-profit first.
  * ``prob_sl`` - calibrated probability of hitting stop-loss first.
  * ``net_reward`` - gross USD reward at TP *minus* all transaction costs.
  * ``net_risk``   - gross USD loss at SL *plus* all transaction costs.

Costs (fees + funding + slippage) are applied so that EV is genuinely net.
"""

from __future__ import annotations

from ae_brain.contracts import EVResult, Side
from ae_brain.risk.costs import CostModel, TradeCosts


class EVGate:
    """Computes net Expected Value and the binary go/no-go decision."""

    def __init__(self, cost_model: CostModel, min_ev_usd: float = 0.0) -> None:
        self._costs = cost_model
        self._min_ev_usd = min_ev_usd

    @staticmethod
    def _normalize_probs(prob_tp: float, prob_sl: float) -> tuple[float, float]:
        """Clamp to [0,1]; if tp+sl > 1 (overlapping), renormalize proportionally.

        prob_tp and prob_sl are competing-risk (first-passage) probabilities, so
        they should sum to <= 1 (the remainder is "neither hit / timeout").
        """
        prob_tp = min(max(prob_tp, 0.0), 1.0)
        prob_sl = min(max(prob_sl, 0.0), 1.0)
        total = prob_tp + prob_sl
        if total > 1.0:
            prob_tp /= total
            prob_sl /= total
        return prob_tp, prob_sl

    def evaluate(
        self,
        *,
        side: Side,
        entry: float,
        take_profit: float,
        stop_loss: float,
        notional_usd: float,
        prob_tp: float,
        prob_sl: float,
        funding_rate_8h: float | None = None,
        holding_hours: float = 8.0,
        adv_usd: float | None = None,
        precomputed_costs: TradeCosts | None = None,
    ) -> EVResult:
        """Return the full :class:`EVResult` (does not mutate state)."""
        prob_tp, prob_sl = self._normalize_probs(prob_tp, prob_sl)

        # Gross reward/risk in USD from price distances scaled to notional.
        qty = notional_usd / entry if entry > 0 else 0.0
        if side == Side.LONG:
            gross_reward = max(take_profit - entry, 0.0) * qty
            gross_risk = max(entry - stop_loss, 0.0) * qty
        else:
            gross_reward = max(entry - take_profit, 0.0) * qty
            gross_risk = max(stop_loss - entry, 0.0) * qty

        costs = precomputed_costs or self._costs.estimate(
            notional_usd,
            side,
            funding_rate_8h=funding_rate_8h,
            holding_hours=holding_hours,
            adv_usd=adv_usd,
        )
        total_cost = costs.total

        # Net of costs: reward shrinks, risk grows.
        net_reward = gross_reward - total_cost
        net_risk = gross_risk + total_cost

        # ---- THE MANDATED EV FORMULA (do not alter) ----------------------
        expected_value = (prob_tp * net_reward) - (prob_sl * net_risk)
        is_positive_ev = expected_value > 0
        # ------------------------------------------------------------------

        # Apply a configurable noise floor on top of the strict >0 rule.
        if is_positive_ev and expected_value < self._min_ev_usd:
            is_positive_ev = False

        return EVResult(
            expected_value=round(expected_value, 6),
            is_positive_ev=bool(is_positive_ev),
            prob_tp=round(prob_tp, 6),
            prob_sl=round(prob_sl, 6),
            net_reward=round(net_reward, 6),
            net_risk=round(net_risk, 6),
            gross_reward=round(gross_reward, 6),
            gross_risk=round(gross_risk, 6),
            total_cost_usd=round(total_cost, 6),
        )
