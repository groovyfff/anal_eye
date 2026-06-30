from __future__ import annotations

from typing import Any

# Input fields that must NOT be treated as ready-made AI answers (stripped before analysis).
_FORBIDDEN_ANSWER_FIELDS = frozenset({
    'decision',
    'signal_type',
    'side',
    'heuristic_signal_consensus',
    'reason_summary',
    'entry_price',
    'tp_price',
    'sl_price',
    'leverage',
    'tp',
    'sl',
    'reason',
    'confidence',
    'consensus_achieved',
})

_DIRECTION_HINT_FIELDS = ('decision', 'signal_type', 'side', 'heuristic_signal_consensus')

_CANDLE_LIST_KEYS = ('candles', 'ohlcv', 'klines', 'kline_data', 'historical_data', 'price_history', 'historical_ohlcv')

_CANDLE_FIELD_ALIASES = {
    'timestamp': ('timestamp', 'time', 'open_time', 'ts'),
    'open': ('open', 'o'),
    'high': ('high', 'h'),
    'low': ('low', 'l'),
    'close': ('close', 'c'),
    'volume': ('volume', 'v'),
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_direction_hint(candidate: dict[str, Any]) -> str | None:
    """Optional weak hint only — not used as final decision."""
    for key in _DIRECTION_HINT_FIELDS:
        raw = candidate.get(key)
        if raw is None:
            continue
        value = str(raw).strip().upper()
        if value in {'LONG', 'SHORT'}:
            return value
    return None


def _pick_candle_field(candle: dict[str, Any], field: str) -> Any:
    for alias in _CANDLE_FIELD_ALIASES[field]:
        if alias in candle and candle[alias] is not None:
            return candle[alias]
    return None


def _normalize_single_candle(candle: dict[str, Any]) -> dict[str, Any]:
    return {
        'timestamp': _pick_candle_field(candle, 'timestamp'),
        'open': _pick_candle_field(candle, 'open'),
        'high': _pick_candle_field(candle, 'high'),
        'low': _pick_candle_field(candle, 'low'),
        'close': _pick_candle_field(candle, 'close'),
        'volume': _pick_candle_field(candle, 'volume'),
    }


def _extract_candles(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    for key in _CANDLE_LIST_KEYS:
        raw = candidate.get(key)
        if isinstance(raw, list) and raw:
            normalized: list[dict[str, Any]] = []
            for item in raw:
                if isinstance(item, dict):
                    normalized.append(_normalize_single_candle(item))
            return normalized
    return []


def _strip_answer_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _FORBIDDEN_ANSWER_FIELDS}


def normalize_candidate(raw: Any) -> dict[str, Any]:
    """Normalize exchange/converter payload; strip pre-filled trading answers."""
    if isinstance(raw, list):
        return {
            '_invalid_payload': True,
            '_invalid_reason': 'payload_is_raw_candles_array_expected_candidate_object',
        }

    if not isinstance(raw, dict):
        return {
            '_invalid_payload': True,
            '_invalid_reason': f'payload_must_be_object_got_{type(raw).__name__}',
        }

    candidate = _strip_answer_fields(dict(raw))
    features: dict[str, Any] = dict(candidate.get('features') or {})

    for key, value in candidate.items():
        if key.startswith('feat_') and value is not None:
            features[key.removeprefix('feat_')] = value

    current_price = (
        _as_float(candidate.get('current_price'))
        or _as_float(features.get('current_price'))
        or _as_float(candidate.get('feat_current_price'))
        or _as_float(candidate.get('price'))
    )
    if current_price is not None:
        candidate['current_price'] = current_price
        features['current_price'] = current_price

    composite = _as_float(candidate.get('composite_score')) or _as_float(
        candidate.get('composite_score_value')
    )
    if composite is not None:
        candidate['composite_score'] = composite

    market_state = candidate.get('market_state') or candidate.get('feat_market_state')
    if market_state is not None:
        candidate['market_state'] = market_state
        features.setdefault('market_state', market_state)

    direction_hint = _extract_direction_hint(raw)
    candidate['direction_hint'] = direction_hint
    candidate['heuristic_signal_consensus'] = direction_hint if direction_hint else 'UNKNOWN'

    candles = _extract_candles(candidate)
    if candles:
        candidate['candles'] = candles
        candidate['historical_ohlcv'] = candles

    candidate['features'] = features
    return candidate


def normalized_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    features = candidate.get('features') or {}
    return {
        'symbol': candidate.get('symbol'),
        'asset_class': candidate.get('asset_class'),
        'source': candidate.get('source'),
        'market': candidate.get('market'),
        'timeframe': candidate.get('timeframe'),
        'direction_hint': candidate.get('direction_hint'),
        'heuristic_signal_consensus': candidate.get('heuristic_signal_consensus'),
        'composite_score': candidate.get('composite_score'),
        'current_price': candidate.get('current_price') or features.get('current_price'),
        'market_state': candidate.get('market_state'),
        'candles_count': len(candidate.get('candles') or []),
        'signal_id': candidate.get('signal_id'),
        'signal_log_db_id': candidate.get('signal_log_db_id'),
    }
