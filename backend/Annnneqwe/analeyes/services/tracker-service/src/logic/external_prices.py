from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EXTERNAL_ASSET_CLASSES = frozenset({'stock', 'metal', 'forex', 'index'})


@dataclass(slots=True)
class PriceQuote:
    symbol: str
    asset_class: str
    price: float
    ts_ms: int
    bid: float | None = None
    ask: float | None = None
    timestamp: dt.datetime | None = None


class ExternalPriceStore:
    """Кэш внешних котировок из data.live_prices.external (ключ — symbol)."""

    def __init__(self, max_age_ms: int = 4500) -> None:
        self.max_age_ms = max_age_ms
        self._by_symbol: dict[str, PriceQuote] = {}
        self._crypto_by_symbol: dict[str, PriceQuote] = {}

    @staticmethod
    def _parse_timestamp(payload: dict[str, Any]) -> tuple[int, dt.datetime | None]:
        ts_ms_raw = payload.get('ts')
        if ts_ms_raw is not None:
            ts_ms = int(ts_ms_raw)
            return ts_ms, dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)
        timestamp_raw = payload.get('timestamp')
        if isinstance(timestamp_raw, str):
            parsed = dt.datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return int(parsed.timestamp() * 1000), parsed
        if isinstance(timestamp_raw, dt.datetime):
            if timestamp_raw.tzinfo is None:
                timestamp_raw = timestamp_raw.replace(tzinfo=dt.timezone.utc)
            return int(timestamp_raw.timestamp() * 1000), timestamp_raw
        now = dt.datetime.now(tz=dt.timezone.utc)
        return int(now.timestamp() * 1000), now

    def _is_fresh(self, ts_ms: int, now_utc: dt.datetime | None = None) -> bool:
        now = now_utc or dt.datetime.now(tz=dt.timezone.utc)
        age_ms = int(now.timestamp() * 1000) - ts_ms
        return age_ms <= self.max_age_ms

    def upsert_external_message(self, payload: dict[str, Any], now_utc: dt.datetime | None = None) -> bool:
        """Обновляет map по сообщению RabbitMQ; возвращает False если котировка устарела."""
        symbol = str(payload.get('symbol', '')).strip()
        if not symbol:
            logger.warning('[tracker] Пропуск live price: пустой symbol')
            return False
        asset_class = str(payload.get('asset_class', 'stock')).strip().lower()
        price_raw = payload.get('price')
        if price_raw is None:
            logger.warning('[tracker] Пропуск live price: нет price symbol=%s', symbol)
            return False
        ts_ms, timestamp = self._parse_timestamp(payload)
        if not self._is_fresh(ts_ms, now_utc=now_utc):
            logger.debug('[tracker] Устаревшая котировка symbol=%s age_ms=%s', symbol, int((now_utc or dt.datetime.now(tz=dt.timezone.utc)).timestamp() * 1000) - ts_ms)
            return False
        quote = PriceQuote(
            symbol=symbol,
            asset_class=asset_class,
            price=float(price_raw),
            ts_ms=ts_ms,
            bid=float(payload['bid']) if payload.get('bid') is not None else None,
            ask=float(payload['ask']) if payload.get('ask') is not None else None,
            timestamp=timestamp,
        )
        self._by_symbol[symbol] = quote
        logger.debug('[tracker] Обновлена external цена symbol=%s price=%s', symbol, quote.price)
        return True

    def upsert_crypto_price(self, symbol: str, price: float, ts_ms: int | None = None) -> None:
        """Инъекция crypto-цены (Binance-style) для интеграционных тестов."""
        now_ms = int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000)
        resolved_ts = now_ms if ts_ms is None else ts_ms
        self._crypto_by_symbol[symbol] = PriceQuote(
            symbol=symbol,
            asset_class='crypto',
            price=float(price),
            ts_ms=resolved_ts,
            timestamp=dt.datetime.fromtimestamp(resolved_ts / 1000, tz=dt.timezone.utc),
        )

    def get_quote(self, symbol: str, asset_class: str, now_utc: dt.datetime | None = None) -> PriceQuote | None:
        asset_class = asset_class.lower()
        if asset_class == 'crypto':
            quote = self._crypto_by_symbol.get(symbol)
        elif asset_class in EXTERNAL_ASSET_CLASSES:
            quote = self._by_symbol.get(symbol)
        else:
            quote = self._by_symbol.get(symbol) or self._crypto_by_symbol.get(symbol)
        if quote is None:
            return None
        if not self._is_fresh(quote.ts_ms, now_utc=now_utc):
            return None
        return quote

    def build_market_data_map(self, now_utc: dt.datetime | None = None) -> dict[str, dict[str, Any]]:
        """Снимок market_data_map для SignalTracker (только свежие котировки)."""
        now = now_utc or dt.datetime.now(tz=dt.timezone.utc)
        merged: dict[str, PriceQuote] = {}
        merged.update(self._crypto_by_symbol)
        merged.update(self._by_symbol)
        result: dict[str, dict[str, Any]] = {}
        for symbol, quote in merged.items():
            if not self._is_fresh(quote.ts_ms, now_utc=now):
                continue
            result[symbol] = {
                'symbol': quote.symbol,
                'asset_class': quote.asset_class,
                'price': quote.price,
                'markPrice': quote.price,
                'bid': quote.bid,
                'ask': quote.ask,
                'ts': quote.ts_ms,
                'timestamp': (quote.timestamp or dt.datetime.fromtimestamp(quote.ts_ms / 1000, tz=dt.timezone.utc)).isoformat().replace('+00:00', 'Z'),
            }
        return result
