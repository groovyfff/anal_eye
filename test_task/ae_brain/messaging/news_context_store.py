"""In-memory store of recent ``news.market_signal`` signals, per symbol.

AE Brain consumes ``news.market_signal`` messages from RabbitMQ and caches the
per-symbol signals here. The fusion layer reads an **aggregated** view (weighted
average of score / relevance / confidence, with linear age decay) to make a
bounded, optional nudge to confidence/EV.

Design
------
* TTL-bounded: signals older than ``ttl_s`` (default 21600s = 6h) expire and are
  pruned on read. An empty store ⇒ neutral / no news ⇒ AE Brain behaves exactly
  as before.
* Age decay: each active signal contributes with weight
  ``relevance × confidence × (1 - age/ttl)`` (linear decay to zero at TTL).
* This module imports only stdlib — no aio_pika, no pydantic — so it is trivially
  unit-testable and never becomes a runtime dependency hazard.

Diagnostic logs emitted here:
* ``news_signal_cached``   — a signal was stored.
* ``news_signal_rejected`` — an invalid signal was dropped.
* ``news_context_expired`` — a signal was pruned because it exceeded the TTL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger("ae_brain.news_context")


@dataclass
class StoredSignal:
    """A single normalized per-symbol news signal held in the cache."""

    symbol: str
    score: int          # 1..10
    relevance: float    # 0..1
    confidence: float   # 0..1
    horizon: str
    source_type: str
    reason: str
    risk_flags: List[str]
    stored_at: float    # monotonic timestamp


@dataclass
class NewsAggregate:
    """Weighted aggregate of all active signals for one symbol.

    ``weight`` is the sum of per-signal weights (relevance × confidence × decay).
    A zero ``signal_count`` means "no active news" — the caller must treat this
    as neutral (no adjustment).
    """

    symbol: str
    score_avg: float       # weighted average score (1..10)
    relevance_avg: float   # weighted average relevance
    confidence_avg: float  # weighted average confidence
    news_strength: float   # abs(bias) × relevance × confidence (after weighting)
    signal_count: int
    weight: float

    @property
    def has_news(self) -> bool:
        return self.signal_count > 0


def score_to_bias(score: float) -> float:
    """Map a 1-10 sentiment score to a [-1, 1] bias.

    ``news_bias = (score - 5.5) / 4.5`` per the spec:
    1 → -1.00, 5 → -0.11, 6 → +0.11, 10 → +1.00.
    """
    return (float(score) - 5.5) / 4.5


class NewsContextStore:
    """Per-symbol, TTL-bounded cache of recent news market signals."""

    def __init__(self, ttl_s: float = 21600.0) -> None:
        # Floor at a tiny epsilon (not 1.0) so tests can exercise short TTLs;
        # production uses the 21600s default. Negative/zero is clamped to the floor.
        self._ttl = max(0.001, float(ttl_s))
        self._store: Dict[str, List[StoredSignal]] = {}

    # --- write ---------------------------------------------------------------

    def add_signal_dict(self, symbol: str, sig: Dict[str, Any]) -> bool:
        """Validate + store one signal dict (from a ``news.market_signal``).

        Returns True if stored, False if rejected (invalid). Emits
        ``news_signal_rejected`` on rejection.
        """
        try:
            score = int(sig.get("score"))
            relevance = float(sig.get("relevance"))
            confidence = float(sig.get("confidence"))
        except (TypeError, ValueError):
            log.info("news_signal_rejected", extra={"event": "news_signal_rejected",
                     "symbol": symbol, "reason": "invalid_types"})
            return False
        if not (1 <= score <= 10) or not (0.0 <= relevance <= 1.0) or not (0.0 <= confidence <= 1.0):
            log.info("news_signal_rejected", extra={"event": "news_signal_rejected",
                     "symbol": symbol, "reason": "out_of_range",
                     "score": score, "relevance": relevance, "confidence": confidence})
            return False
        sym = str(symbol or "").strip().upper()
        if not sym:
            log.info("news_signal_rejected", extra={"event": "news_signal_rejected",
                     "symbol": symbol, "reason": "missing_symbol"})
            return False
        stored = StoredSignal(
            symbol=sym,
            score=score,
            relevance=relevance,
            confidence=confidence,
            horizon=str(sig.get("horizon", "medium")),
            source_type=str(sig.get("source_type", "other")),
            reason=str(sig.get("reason", "")),
            risk_flags=list(sig.get("risk_flags", []) or []),
            stored_at=time.monotonic(),
        )
        self._store.setdefault(sym, []).append(stored)
        log.info("news_signal_cached", extra={"event": "news_signal_cached",
                 "symbol": sym, "score": score, "relevance": relevance,
                 "confidence": confidence})
        return True

    # --- read ----------------------------------------------------------------

    def _prune(self, symbol: str, now: float) -> List[StoredSignal]:
        """Drop expired signals for ``symbol``; return the active ones."""
        entries = self._store.get(symbol, [])
        if not entries:
            return []
        active: List[StoredSignal] = []
        for s in entries:
            age = now - s.stored_at
            if age >= self._ttl:
                log.info("news_context_expired", extra={"event": "news_context_expired",
                         "symbol": symbol, "age_s": age, "ttl_s": self._ttl})
                continue
            active.append(s)
        self._store[symbol] = active
        return active

    def get_active(self, symbol: str) -> List[StoredSignal]:
        """Return non-expired signals for ``symbol`` (prunes as a side effect)."""
        return self._prune(str(symbol or "").strip().upper(), time.monotonic())

    def aggregate(self, symbol: str) -> NewsAggregate:
        """Compute the weighted aggregate for ``symbol``.

        Weight per signal = ``relevance × confidence × (1 - age/ttl)`` (linear
        age decay). If no active signals, returns an aggregate with
        ``signal_count == 0`` (neutral).
        """
        sym = str(symbol or "").strip().upper()
        active = self._prune(sym, time.monotonic())
        if not active:
            return NewsAggregate(symbol=sym, score_avg=5.5, relevance_avg=0.0,
                                 confidence_avg=0.0, news_strength=0.0,
                                 signal_count=0, weight=0.0)

        now = time.monotonic()
        w_score = 0.0
        w_relevance = 0.0
        w_confidence = 0.0
        # Weighted strength: average of abs(bias)*relevance*confidence weighted
        # by (relevance*confidence*decay) so stronger/fresher signals dominate.
        w_strength = 0.0
        weight_sum = 0.0
        for s in active:
            decay = max(0.0, 1.0 - (now - s.stored_at) / self._ttl)
            w = s.relevance * s.confidence * decay
            if w <= 0.0:
                continue
            bias_abs = abs(score_to_bias(s.score))
            w_score += s.score * w
            w_relevance += s.relevance * w
            w_confidence += s.confidence * w
            w_strength += bias_abs * s.relevance * s.confidence * decay
            weight_sum += w

        if weight_sum <= 0.0:
            return NewsAggregate(symbol=sym, score_avg=5.5, relevance_avg=0.0,
                                 confidence_avg=0.0, news_strength=0.0,
                                 signal_count=0, weight=0.0)

        return NewsAggregate(
            symbol=sym,
            score_avg=w_score / weight_sum,
            relevance_avg=w_relevance / weight_sum,
            confidence_avg=w_confidence / weight_sum,
            # strength is the weighted mean of per-signal strength.
            news_strength=(w_strength / weight_sum),
            signal_count=len(active),
            weight=weight_sum,
        )

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return sum(len(v) for v in self._store.values())
