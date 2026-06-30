"""Shared domain contracts (typed messages passed between layers).

Keeping these as small frozen dataclasses (rather than ad-hoc dicts) gives us a
typed seam between the RabbitMQ boundary, the inference engine, and the fusion
layer, and makes the final ``signal.final`` payload deterministic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ae_brain.symbols import extract_base_asset, require_symbol


class Decision(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    SKIP = "SKIP"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class AssetClass(str, Enum):
    """Asset classes supported by the corporate backend (PRD/ТЗ #2)."""

    CRYPTO = "crypto"
    STOCK = "stock"
    METAL = "metal"
    FOREX = "forex"


#: Asset classes that *do* carry perpetual-derivatives microstructure (funding,
#: open interest, CVD, liquidations). Everything else is a "traditional" asset
#: for which those fields arrive as ``null`` and are mapped to neutral defaults.
DERIVATIVE_ASSET_CLASSES: frozenset[str] = frozenset({AssetClass.CRYPTO.value})

#: Minimum candle window the sequence layer needs (see ModelConfig.sequence_window).
MIN_SEQUENCE_CANDLES: int = 48


@dataclass(slots=True)
class TradeCandidate:
    """Inbound message consumed from ``data.candidates.ai``.

    Carries the symbol + the raw candle window (most recent last) and any
    precomputed microstructure context. Features are (re)derived deterministically
    on our side to guarantee the train/serve feature contract.

    Backend integration (PRD/ТЗ #2)
    -------------------------------
    * ``signal_log_db_id`` - the id of the row the backend already INSERTed into
      ``signal_feature_logs``. The ensemble writes its outputs back into *that*
      row via UPDATE (no second INSERT), giving the backend a stable handle.
    * ``asset_class`` - one of crypto / stock / metal / forex. Drives null-handling
      for derivatives-only microstructure fields.
    """

    symbol: str
    interval: str  # e.g. "5m"
    candles: list[dict[str, Any]]  # OHLCV(+microstructure) rows, oldest -> newest
    signal_log_db_id: int  # pre-inserted backend row id (mandatory)
    asset_class: str = AssetClass.CRYPTO.value
    signal_id: str = ""  # backend candidate uuid (propagated to signal.final)
    correlation_id: str = ""
    received_ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce/validate asset_class to the known set; default to crypto.
        try:
            self.asset_class = AssetClass(str(self.asset_class).lower()).value
        except ValueError:
            self.asset_class = AssetClass.CRYPTO.value

    @property
    def is_derivative(self) -> bool:
        """True if this asset carries funding/OI/CVD/liquidation microstructure."""
        return self.asset_class in DERIVATIVE_ASSET_CLASSES

    @staticmethod
    def _normalize_candles(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize candle rows so the feature engineer always finds a ``ts``.

        The backend producer keys the candle time as ``timestamp``; the crypto
        producer uses ``open_time``. Both are mapped to ``ts`` (used for the
        session sin/cos time features) without mutating the source dicts.
        """
        normalized: list[dict[str, Any]] = []
        for row in raw:
            candle = dict(row)
            if "ts" not in candle:
                ts_value = candle.get("timestamp", candle.get("open_time"))
                if ts_value is not None:
                    candle["ts"] = ts_value
            normalized.append(candle)
        return normalized

    @property
    def asset(self) -> str:
        """Base asset derived from :attr:`symbol` (e.g. ``ETH`` from ``ETHUSDT``)."""
        return extract_base_asset(self.symbol)

    @classmethod
    def from_message(cls, payload: dict[str, Any]) -> "TradeCandidate":
        symbol = require_symbol(payload.get("symbol"))
        # Backward compatibility: legacy crypto producers publish without a
        # ``signal_log_db_id``. A missing or null id defaults to 0, which routes
        # the engine to the INSERT fallback instead of the backend UPDATE path.
        raw_id = payload.get("signal_log_db_id")
        signal_log_db_id = int(raw_id) if raw_id is not None else 0
        # The sequence layer's candle window arrives under ``candles`` (crypto
        # producer) or ``historical_ohlcv`` (external-markets backend producer).
        raw_candles = payload.get("candles")
        if raw_candles is None:
            for key in ("ohlcv", "klines", "kline_data", "historical_data", "price_history", "historical_ohlcv"):
                if payload.get(key):
                    raw_candles = payload[key]
                    break
        if raw_candles is None:
            raw_candles = []
        meta = dict(payload.get("meta", {}))
        if payload.get("features"):
            meta.setdefault("features", dict(payload["features"]))
        if payload.get("composite_score") is not None:
            meta.setdefault("composite_score", payload["composite_score"])
        if payload.get("current_price") is not None:
            meta.setdefault("current_price", payload["current_price"])
        return cls(
            symbol=symbol,
            interval=str(payload.get("interval", payload.get("timeframe", "5m"))),
            candles=cls._normalize_candles(list(raw_candles)),
            signal_log_db_id=signal_log_db_id,
            asset_class=str(payload.get("asset_class", AssetClass.CRYPTO.value)),
            signal_id=str(payload.get("signal_id", "")),
            correlation_id=str(payload.get("correlation_id", "")),
            meta=meta,
        )


@dataclass(slots=True)
class LayerProbabilities:
    """Calibrated probabilities emitted by the predictive layers."""

    # Tabular: P(take-profit hit before stop) for the long-direction view.
    tabular_p_up: float = 0.5
    # Sequence: P(trend continuation) and reversal is its complement.
    sequence_p_continuation: float = 0.5
    sequence_trend_sign: float = 0.0  # +1 up-trend, -1 down-trend, 0 flat
    # RL agent suggested action in [-1, 1] (signed target exposure) + value.
    rl_target_exposure: float = 0.0
    rl_state_value: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class EVResult:
    """Output of the Expected-Value gate."""

    expected_value: float
    is_positive_ev: bool
    prob_tp: float
    prob_sl: float
    net_reward: float
    net_risk: float
    gross_reward: float
    gross_risk: float
    total_cost_usd: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FinalSignal:
    """Strict, deterministic output published to ``signal.final``."""

    symbol: str
    decision: Decision
    position_size_pct: float  # fraction of account equity (0..max_position_pct)
    leverage: float
    take_profit: float  # absolute price
    stop_loss: float  # absolute price
    entry_reference: float  # reference/mark price used for sizing
    expected_value_usd: float
    confidence: float  # fused directional conviction in [0, 1]
    correlation_id: str = ""
    # Identity / routing context propagated from the candidate so the published
    # ``signal.final`` message satisfies the tracker-service contract.
    signal_id: str = ""
    asset_class: str = ""
    signal_log_db_id: int = 0
    source_ai: str = "ae_brain"
    ev: dict[str, Any] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    version: str = "0.1.0"

    @property
    def asset(self) -> str:
        """Base asset derived from :attr:`symbol` (e.g. ``ETH`` from ``ETHUSDT``)."""
        return extract_base_asset(self.symbol)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["asset"] = self.asset
        # --- tracker-service bridge (signal.final contract) ----------------
        # SignalTracker.start_tracking_signal reads tp/sl/entry_price/signal_id/
        # asset_class/signal_log_db_id/source_ai/signal_time. Expose them here
        # so the ensemble's output is consumable without a separate adapter.
        d["tp"] = self.take_profit
        d["sl"] = self.stop_loss
        d["entry_price"] = self.entry_reference
        d["signal_type"] = self.decision.value
        d["reason_summary"] = (
            f"AE Brain {self.components.get('decision_source', 'fusion')}: "
            f"{self.decision.value} on {self.symbol} "
            f"(asset={self.asset}, confidence={self.confidence:.3f})"
        )
        d["tp_price"] = self.take_profit
        d["sl_price"] = self.stop_loss
        d["consensus_achieved"] = self.decision.value in ("LONG", "SHORT")
        d["signal_time"] = (
            datetime.fromtimestamp(self.ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return d
