"""Layer 4 - Fusion / Output layer.

Aggregates the calibrated outputs of the three predictive layers into a single,
strict, deterministic decision. The pipeline is:

1. Convert each layer's output into a directional signal in ``[-1, 1]``.
2. Weighted-blend into a fused conviction + direction.
3. Translate calibrated probabilities into first-passage ``prob_tp`` / ``prob_sl``
   for the chosen side.
4. Place ATR-based TP/SL and compute fractional-Kelly / vol-targeted size,
   respecting the correlation budget.
5. Run the **EV gate**. Publish LONG/SHORT only if EV > 0 *and* conviction and
   size pass their floors; otherwise SKIP.

Output is a :class:`~ae_brain.contracts.FinalSignal` -> deterministic JSON dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ae_brain.config import FusionConfig, RiskConfig
from ae_brain.contracts import (
    Decision,
    EVResult,
    FinalSignal,
    LayerProbabilities,
    Side,
)
from ae_brain.risk.ev_gate import EVGate
from ae_brain.risk.sizing import PositionSizer
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.fusion")


@dataclass(slots=True)
class FusionContext:
    """Market context required to turn probabilities into a sized, priced order."""

    symbol: str
    entry_price: float
    atr: float
    funding_rate_8h: float = 0.0
    adv_usd: float | None = None
    holding_hours: float = 8.0
    correlated_exposure: float = 0.0
    correlation_id: str = ""
    # Current market-regime one-hot [trend, chop, highvol] for the meta-model.
    regime_onehot: tuple[float, float, float] = (0.0, 0.0, 0.0)


class FusionLayer:
    def __init__(
        self,
        fusion_cfg: FusionConfig,
        risk_cfg: RiskConfig,
        ev_gate: EVGate,
        sizer: PositionSizer,
        meta_model: object | None = None,
    ) -> None:
        self._cfg = fusion_cfg
        self._risk = risk_cfg
        self._ev = ev_gate
        self._sizer = sizer
        # Optional trained meta-classifier. When ready, it is the directional
        # decision authority (replacing the heuristic conviction+EV gate).
        self._meta = meta_model

    # ------------------------------------------------------------------ #
    @staticmethod
    def _directional_signals(p: LayerProbabilities) -> tuple[float, float, float]:
        """Map each layer to a directional signal in [-1, 1]."""
        tab_dir = 2.0 * p.tabular_p_up - 1.0
        # Sequence continuation only matters in the direction of the trend sign.
        seq_dir = p.sequence_trend_sign * (2.0 * p.sequence_p_continuation - 1.0)
        rl_dir = float(np.clip(p.rl_target_exposure, -1.0, 1.0))
        return tab_dir, seq_dir, rl_dir

    def _fuse(self, p: LayerProbabilities) -> tuple[float, float]:
        """Return (fused_score in [-1,1], conviction in [0,1])."""
        tab_dir, seq_dir, rl_dir = self._directional_signals(p)
        w = self._cfg
        wsum = w.w_tabular + w.w_sequence + w.w_rl
        fused = (w.w_tabular * tab_dir + w.w_sequence * seq_dir + w.w_rl * rl_dir) / wsum
        fused = float(np.clip(fused, -1.0, 1.0))
        return fused, abs(fused)

    @staticmethod
    def _side_probabilities(p: LayerProbabilities, side: Side) -> tuple[float, float]:
        """Blend calibrated layer probs into first-passage prob_tp / prob_sl.

        For a LONG, prob_tp is driven by P(up) and P(continuation); for a SHORT
        it is their directional complement. prob_sl is the residual adverse mass
        (kept < 1-prob_tp so the EV gate sees a proper competing-risk pair).
        """
        p_up = p.tabular_p_up
        cont = p.sequence_p_continuation
        # Align sequence continuation prob with the requested side.
        if (p.sequence_trend_sign >= 0 and side == Side.LONG) or (
            p.sequence_trend_sign < 0 and side == Side.SHORT
        ):
            seq_favor = cont
        else:
            seq_favor = 1.0 - cont

        favor_long = 0.6 * p_up + 0.4 * seq_favor
        prob_tp = favor_long if side == Side.LONG else 1.0 - favor_long
        prob_tp = float(np.clip(prob_tp, 0.01, 0.99))
        # Adverse mass: the rest, minus a "neither/timeout" allowance.
        prob_sl = float(np.clip((1.0 - prob_tp) * 0.85, 0.01, 0.99))
        return prob_tp, prob_sl

    # ------------------------------------------------------------------ #
    def _predict_meta(self, probs: LayerProbabilities, ctx: FusionContext):
        """Run the meta-model and return raw class probabilities."""
        if self._meta is None or not getattr(self._meta, "is_ready", lambda: False)():
            return None
        from ae_brain.layers.meta import build_meta_features

        mf = build_meta_features(
            probs.tabular_p_up,
            probs.sequence_p_continuation,
            probs.sequence_trend_sign,
            probs.rl_target_exposure,
            ctx.regime_onehot,
        )
        return self._meta.predict(mf)

    @staticmethod
    def _risk_approves(sizing, ev: EVResult) -> bool:
        """Sizing + EV must both pass before a threshold-qualified meta trade fires."""
        return (
            sizing.kelly_fraction_raw > 0.0
            and sizing.position_size_pct > 0.0
            and sizing.rejected_reason is None
            and ev.is_positive_ev
        )

    def decide(self, probs: LayerProbabilities, ctx: FusionContext) -> FinalSignal:
        """Produce the final decision.

        When a trained meta-model is attached it supplies directional confidence
        via ``meta_direction_threshold``; the risk engine (Kelly + EV) decides
        go/no-go. Otherwise the legacy conviction + EV gate applies.
        """
        from ae_brain.layers.meta import CLASS_LONG, CLASS_SHORT, resolve_directional_class

        fused, conviction = self._fuse(probs)
        heuristic_side = Side.LONG if fused >= 0 else Side.SHORT
        threshold = self._cfg.meta_direction_threshold

        meta_pred = self._predict_meta(probs, ctx)
        directional_class = None
        if meta_pred is not None:
            directional_class = resolve_directional_class(
                meta_pred.p_short,
                meta_pred.p_long,
                threshold=threshold,
            )
            if directional_class == CLASS_LONG:
                side = Side.LONG
                conviction = meta_pred.p_long
            elif directional_class == CLASS_SHORT:
                side = Side.SHORT
                conviction = meta_pred.p_short
            else:
                side = heuristic_side
                conviction = meta_pred.p_skip
        else:
            side = heuristic_side

        # Reward:risk ratio implied by ATR multiples.
        rr_ratio = self._risk.atr_tp_mult / max(self._risk.atr_sl_mult, 1e-9)
        prob_tp, prob_sl = self._side_probabilities(probs, side)

        sizing = self._sizer.size(
            entry=ctx.entry_price,
            atr=ctx.atr,
            side=side,
            prob_tp=prob_tp,
            reward_risk_ratio=rr_ratio,
            correlated_exposure=ctx.correlated_exposure,
        )

        ev: EVResult = self._ev.evaluate(
            side=side,
            entry=ctx.entry_price,
            take_profit=sizing.take_profit,
            stop_loss=sizing.stop_loss,
            notional_usd=sizing.notional_usd,
            prob_tp=prob_tp,
            prob_sl=prob_sl,
            funding_rate_8h=ctx.funding_rate_8h,
            holding_hours=ctx.holding_hours,
            adv_usd=ctx.adv_usd,
        )

        if meta_pred is not None:
            if directional_class is None or not self._risk_approves(sizing, ev):
                decision = Decision.SKIP
            else:
                decision = Decision.LONG if side == Side.LONG else Decision.SHORT
        else:
            decision = self._gate_decision(side, conviction, sizing, ev)

        components = {
            "fused_score": round(fused, 6),
            "directional": dict(
                zip(("tabular", "sequence", "rl"), self._directional_signals(probs))
            ),
            "layer_probs": probs.as_dict(),
            "regime_onehot": list(ctx.regime_onehot),
            "decision_source": "meta_model" if meta_pred is not None else "heuristic_ev_gate",
            "meta": (
                {
                    "p_short": round(meta_pred.p_short, 6),
                    "p_skip": round(meta_pred.p_skip, 6),
                    "p_long": round(meta_pred.p_long, 6),
                    "direction_threshold": round(threshold, 4),
                    "directional_class": directional_class,
                    "long_passes_threshold": meta_pred.p_long > threshold,
                    "short_passes_threshold": meta_pred.p_short > threshold,
                }
                if meta_pred is not None
                else None
            ),
            "sizing": {
                "kelly_fraction_raw": sizing.kelly_fraction_raw,
                "correlation_scale": sizing.correlation_scale,
                "notional_usd": sizing.notional_usd,
                "rejected_reason": sizing.rejected_reason,
                "reward_risk_ratio": round(rr_ratio, 4),
            },
        }

        log.info(
            "fusion.decision_debug",
            symbol=ctx.symbol,
            decision=str(decision),
            p_short=round(meta_pred.p_short, 6) if meta_pred else None,
            p_skip=round(meta_pred.p_skip, 6) if meta_pred else None,
            p_long=round(meta_pred.p_long, 6) if meta_pred else None,
            meta_direction_threshold=threshold,
            directional_class=directional_class,
            kelly_fraction=sizing.kelly_fraction_raw,
            position_size_pct=sizing.position_size_pct,
            ev_usd=ev.expected_value,
            is_positive_ev=ev.is_positive_ev,
            risk_approves=self._risk_approves(sizing, ev),
            sizing_rejected=sizing.rejected_reason,
            decision_source=components["decision_source"],
        )

        return FinalSignal(
            symbol=ctx.symbol,
            decision=decision,
            position_size_pct=sizing.position_size_pct if decision != Decision.SKIP else 0.0,
            leverage=sizing.leverage if decision != Decision.SKIP else 0.0,
            take_profit=round(sizing.take_profit, 8) if decision != Decision.SKIP else 0.0,
            stop_loss=round(sizing.stop_loss, 8) if decision != Decision.SKIP else 0.0,
            entry_reference=round(ctx.entry_price, 8),
            expected_value_usd=ev.expected_value,
            confidence=round(conviction, 6),
            correlation_id=ctx.correlation_id,
            ev=ev.as_dict(),
            components=components,
        )

    def _gate_decision(
        self,
        side: Side,
        conviction: float,
        sizing,
        ev: EVResult,
    ) -> Decision:
        """Apply the strict gates in priority order."""
        if conviction < self._cfg.min_conviction:
            return Decision.SKIP
        if sizing.position_size_pct <= 0.0 or sizing.rejected_reason:
            return Decision.SKIP
        if not ev.is_positive_ev:
            return Decision.SKIP
        return Decision.LONG if side == Side.LONG else Decision.SHORT
