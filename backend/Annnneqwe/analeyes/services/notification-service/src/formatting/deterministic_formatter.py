from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TELEGRAM_CAPTION_LIMIT = 1024

_FINANCIAL_ADVICE_PHRASES = (
    "guaranteed",
    "100%",
    "без риска",
    "гарантирован",
    "гарантия прибыли",
    "безубыточно",
)


@dataclass(frozen=True, slots=True)
class SignalFacts:
    symbol: str
    decision: str
    confidence: float | None
    market_state: str | None
    trend_impulse: Any
    entry_price: Any
    stop_loss: Any
    take_profit: Any
    leverage: Any
    model_name: str
    timestamp_utc: datetime | None
    known_prices: frozenset[str]
    known_confidence_strings: frozenset[str]

    def to_sanitized_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "confidence_percent": self._confidence_percent(),
            "market_state": self.market_state or "N/A",
            "trend_impulse": self.trend_impulse if self.trend_impulse is not None else "N/A",
            "entry_price": self._format_price(self.entry_price),
            "stop_loss": self._format_price(self.stop_loss),
            "take_profit": self._format_price(self.take_profit),
            "leverage": self._format_leverage_value(),
            "model_name": self.model_name,
            "timestamp_utc": self._format_timestamp(),
        }

    def _confidence_percent(self) -> int | None:
        if self.confidence is None:
            return None
        value = float(self.confidence)
        if value > 1.0:
            value = value / 100.0
        return int(round(value * 100))

    @staticmethod
    def _format_price(value: Any) -> str:
        if value is None or value == "" or str(value).lower() == "market":
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_leverage_value(self) -> str:
        if self.leverage is None or self.leverage == "":
            return "N/A"
        try:
            lev = float(self.leverage)
            if lev == int(lev):
                return str(int(lev))
            return f"{lev:g}"
        except (TypeError, ValueError):
            return str(self.leverage)

    def _format_timestamp(self) -> str:
        if self.timestamp_utc is None:
            return "N/A"
        return self.timestamp_utc.strftime("%d.%m.%Y %H:%M:%S UTC")


def extract_signal_facts(payload: dict[str, Any]) -> SignalFacts:
    features = payload.get("features") or {}
    symbol = str(payload.get("symbol") or payload.get("name") or "UNKNOWN").upper()
    decision = str(payload.get("decision") or payload.get("signal_type") or payload.get("side") or "").upper()

    confidence = _parse_confidence(payload.get("confidence"))

    market_state = _first_non_empty(
        payload.get("market_state"),
        features.get("market_state"),
        features.get("feat_market_state"),
    )
    if market_state is not None:
        market_state = str(market_state).upper()

    trend_impulse = _first_non_empty(
        features.get("trend_impulse_10m"),
        features.get("trend_impulse"),
        features.get("impulse"),
        features.get("momentum"),
        payload.get("trend_impulse_10m"),
        payload.get("trend_impulse"),
        payload.get("impulse"),
        payload.get("momentum"),
    )

    entry_price = _first_non_empty(
        payload.get("entry_price"),
        payload.get("entry"),
        payload.get("current_price"),
        payload.get("execution_price"),
    )
    stop_loss = _first_non_empty(payload.get("stop_loss"), payload.get("sl"), payload.get("sl_price"))
    take_profit = _first_non_empty(payload.get("take_profit"), payload.get("tp"), payload.get("tp_price"))
    leverage = _first_non_empty(payload.get("leverage"), payload.get("max_leverage"))

    model_name = str(
        _first_non_empty(
            payload.get("model"),
            payload.get("model_name"),
            payload.get("source_ai"),
            payload.get("ensemble_name"),
        )
        or "AE Brain"
    )

    timestamp_utc = _parse_timestamp(
        payload.get("timestamp"),
        payload.get("event_time"),
        payload.get("signal_time"),
        payload.get("execution_time"),
        payload.get("ts"),
    )

    known_prices = _collect_known_prices(entry_price, stop_loss, take_profit, payload)
    known_confidence_strings = _collect_known_confidence_strings(confidence)

    return SignalFacts(
        symbol=symbol,
        decision=decision,
        confidence=confidence,
        market_state=market_state,
        trend_impulse=trend_impulse,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
        model_name=model_name,
        timestamp_utc=timestamp_utc,
        known_prices=known_prices,
        known_confidence_strings=known_confidence_strings,
    )


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _parse_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(*values: Any) -> datetime | None:
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        raw = str(value).strip()
        if not raw:
            continue
        try:
            if raw.isdigit():
                ts = float(raw)
                if ts > 1e12:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            normalized = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _collect_known_prices(entry: Any, sl: Any, tp: Any, payload: dict[str, Any]) -> frozenset[str]:
    prices: set[str] = set()
    for value in (
        entry,
        sl,
        tp,
        payload.get("execution_price"),
        payload.get("signal_reference_price"),
        payload.get("current_price"),
    ):
        if value is None or value == "" or str(value).lower() == "market":
            continue
        try:
            num = float(value)
            prices.add(f"{num:.2f}")
            prices.add(f"{num:.6g}")
            prices.add(str(num))
        except (TypeError, ValueError):
            prices.add(str(value))
    return frozenset(prices)


def _collect_known_confidence_strings(confidence: float | None) -> frozenset[str]:
    if confidence is None:
        return frozenset()
    value = float(confidence)
    if value > 1.0:
        value = value / 100.0
    percent = int(round(value * 100))
    return frozenset({str(percent), f"{percent}%", f"{value:.2f}", f"{value:.3f}"})


class DeterministicSignalFormatter:
    """Build Telegram captions deterministically from signal.final facts."""

    def build_fallback_comment(self, facts: SignalFacts) -> str:
        parts: list[str] = []
        if facts.decision == "LONG":
            parts.append(
                "Сигнал указывает на преимущество покупателей, "
                "но вход стоит рассматривать только с учётом риска и текущей волатильности."
            )
        elif facts.decision == "SHORT":
            parts.append(
                "Сигнал указывает на давление продавцов. "
                "Движение может продолжиться, но риск резких отскоков остаётся."
            )

        conf = facts._confidence_percent()
        if conf is not None and conf >= 80:
            parts.append(
                "Модель видит сильное подтверждение направления, "
                "однако риск-менеджмент остаётся обязательным."
            )

        if facts.market_state == "TREND":
            parts.append("Сценарий поддерживается трендовым состоянием рынка.")

        missing = any(
            value is None or value == ""
            for value in (facts.entry_price, facts.stop_loss, facts.take_profit, facts.leverage)
        )
        if missing:
            parts.append(
                "Сигнал сформирован на доступных рыночных данных; "
                "часть торговых параметров недоступна."
            )

        if not parts:
            return "Сигнал сформирован на доступных рыночных данных."
        return " ".join(parts)

    def build_caption(self, facts: SignalFacts, comment: str) -> str:
        direction_line = self._direction_line(facts.decision)
        confidence_line = self._confidence_line(facts)
        market_state = facts.market_state or "N/A"
        trend_impulse = facts.trend_impulse if facts.trend_impulse is not None else "N/A"
        entry = facts._format_price(facts.entry_price)
        sl = facts._format_price(facts.stop_loss)
        tp = facts._format_price(facts.take_profit)
        leverage = facts._format_leverage_value()
        timestamp = facts._format_timestamp()
        binance_url = f"https://www.binance.com/en/futures/{facts.symbol}"

        lines = [
            "🚀 AnalEyes Рекомендация",
            f"💎 Монета: {facts.symbol}",
            f"🧠 Модель AI: {facts.model_name}",
            f"🎯 Направление: {direction_line}",
            f"🔥 Уверенность AI: {confidence_line}",
            f"📊 Состояние Рынка: {market_state}",
            f"📈 Тренд-импульс (10m): {trend_impulse}",
            f"📉 Вход (предполагаемый): {entry} USDT",
            f"💰 Стоп-Лосс (SL): {sl} USDT",
            f"🏆 Тейк-Профит (TP): {tp} USDT",
            f"🧮 Плечо (макс): x{leverage}",
            f"💬 Комментарий AI: {comment}",
            f"🔗 Торговать {facts.symbol} на Binance Futures",
            f"🕘 {timestamp}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _direction_line(decision: str) -> str:
        if decision == "LONG":
            return "🟢 Лонг"
        if decision == "SHORT":
            return "🔴 Шорт"
        return decision or "N/A"

    @staticmethod
    def _confidence_line(facts: SignalFacts) -> str:
        percent = facts._confidence_percent()
        if percent is None:
            return "N/A"
        return f"{percent}%"

    def truncate_caption_for_telegram(self, caption: str, *, max_len: int = TELEGRAM_CAPTION_LIMIT) -> str:
        if len(caption) <= max_len:
            return caption
        marker = "💬 Комментарий AI: "
        start = caption.find(marker)
        if start == -1:
            return caption[: max_len - 3] + "..."
        prefix = caption[: start + len(marker)]
        suffix_start = caption.find("\n", start)
        if suffix_start == -1:
            return caption[: max_len - 3] + "..."
        suffix = caption[suffix_start:]
        available = max_len - len(prefix) - len(suffix) - 3
        if available < 20:
            return caption[: max_len - 3] + "..."
        comment = caption[start + len(marker) : suffix_start]
        return prefix + comment[:available] + "..." + suffix


def validate_llm_comment(comment: str, facts: SignalFacts, *, max_chars: int) -> str | None:
    """Return rejection reason or None if comment is acceptable."""
    text = str(comment or "").strip()
    if not text:
        return "empty_comment"
    if len(text) > max_chars:
        return "too_long"

    lower = text.lower()
    for phrase in _FINANCIAL_ADVICE_PHRASES:
        if phrase in lower:
            return "financial_advice"

    if facts.decision == "LONG" and _mentions_short(lower) and not _mentions_long(lower):
        return "contradicts_direction"
    if facts.decision == "SHORT" and _mentions_long(lower) and not _mentions_short(lower):
        return "contradicts_direction"

    for match in re.findall(r"\b[A-Z]{2,10}USDT\b", text.upper()):
        if match != facts.symbol:
            return "foreign_symbol"

    for price in _extract_price_like_tokens(text):
        if price not in facts.known_prices:
            return "fabricated_price"

    for conf in _extract_confidence_like_tokens(text):
        if conf not in facts.known_confidence_strings:
            return "fabricated_confidence"

    return None


def _mentions_long(text: str) -> bool:
    return bool(re.search(r"\b(long|лонг|покупател|быч)\b", text, re.I))


def _mentions_short(text: str) -> bool:
    return bool(re.search(r"\b(short|шорт|продавц|медв)\b", text, re.I))


def _extract_price_like_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.findall(r"\b\d+(?:[.,]\d{2,8})\b", text):
        normalized = match.replace(",", "")
        try:
            num = float(normalized)
            if num > 10:
                tokens.add(f"{num:.2f}")
                tokens.add(f"{num:.6g}")
                tokens.add(str(num))
        except ValueError:
            continue
    return tokens


def _extract_confidence_like_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.findall(r"\b(\d{1,3})\s*%", text):
        tokens.add(match)
        tokens.add(f"{match}%")
    return tokens
