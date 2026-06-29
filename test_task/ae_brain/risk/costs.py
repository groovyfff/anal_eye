"""Transaction cost model for Binance USD-M futures.

All EV math is computed *net* of these costs. We model three components:

1. **Exchange fees** - taker/maker bps on both entry and exit notional.
2. **Funding** - perpetual funding paid/received over the expected holding
   period (8h epochs). Sign depends on side and funding rate.
3. **Slippage** - a square-root market-impact style model: base spread cost
   plus an impact term scaling with participation in average daily volume.

The model returns costs in **USD** so they can be subtracted directly from the
gross reward/risk in the EV gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ae_brain.config import CostConfig
from ae_brain.contracts import Side


@dataclass(slots=True)
class TradeCosts:
    """Itemized round-trip trade costs in USD."""

    fee_usd: float
    funding_usd: float
    slippage_usd: float

    @property
    def total(self) -> float:
        return self.fee_usd + self.funding_usd + self.slippage_usd


class CostModel:
    """Deterministic, parameterized cost estimator."""

    def __init__(self, cfg: CostConfig) -> None:
        self._cfg = cfg

    def fees(self, notional_usd: float, *, taker: bool = True, round_trip: bool = True) -> float:
        rate = self._cfg.taker_fee_rate if taker else self._cfg.maker_fee_rate
        legs = 2 if round_trip else 1
        return abs(notional_usd) * rate * legs

    def funding(
        self,
        notional_usd: float,
        side: Side,
        *,
        funding_rate_8h: float | None = None,
        holding_hours: float = 8.0,
    ) -> float:
        """Expected funding cost (positive = cost to us).

        Longs pay positive funding; shorts receive it (and vice-versa). We
        prorate by the number of 8h funding epochs in the holding window.
        """
        rate = self._cfg.default_funding_rate_8h if funding_rate_8h is None else funding_rate_8h
        epochs = max(holding_hours / 8.0, 0.0)
        signed = rate if side == Side.LONG else -rate
        return abs(notional_usd) * signed * epochs

    def slippage(self, notional_usd: float, *, adv_usd: float | None = None) -> float:
        """Slippage in USD using base spread + sqrt participation impact."""
        base = abs(notional_usd) * (self._cfg.base_slippage_bps / 1e4)
        if adv_usd and adv_usd > 0:
            participation = abs(notional_usd) / adv_usd
            impact_bps = self._cfg.slippage_impact_coeff * math.sqrt(participation) * 1e4
            impact = abs(notional_usd) * (impact_bps / 1e4)
        else:
            impact = abs(notional_usd) * (self._cfg.base_slippage_bps / 1e4)
        # Slippage applies on both entry and exit.
        return (base + impact) * 2

    def estimate(
        self,
        notional_usd: float,
        side: Side,
        *,
        funding_rate_8h: float | None = None,
        holding_hours: float = 8.0,
        adv_usd: float | None = None,
        taker: bool = True,
    ) -> TradeCosts:
        """Return itemized round-trip costs in USD."""
        return TradeCosts(
            fee_usd=self.fees(notional_usd, taker=taker, round_trip=True),
            funding_usd=self.funding(
                notional_usd, side, funding_rate_8h=funding_rate_8h, holding_hours=holding_hours
            ),
            slippage_usd=self.slippage(notional_usd, adv_usd=adv_usd),
        )
