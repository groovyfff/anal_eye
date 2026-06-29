from __future__ import annotations
import datetime as dt
import math
import uuid
from typing import Any

def validate_candidate_payload(payload: dict[str, Any], require_db_id: bool = False) -> list[str]:
    errors: list[str] = []
    required = ['signal_id', 'symbol', 'asset_class', 'timestamp', 'trigger_reason', 'heuristic_signal_consensus', 'features', 'indicators', 'patterns', 'historical_snapshots', 'composite_score', 'entry_price_suggestion', 'signal_log_db_id']
    for key in required:
        if key not in payload:
            errors.append(f'missing field: {key}')
    signal_id = payload.get('signal_id')
    if not _is_uuid_v4(signal_id):
        errors.append('signal_id must be UUID v4')
    symbol = payload.get('symbol')
    if not isinstance(symbol, str) or not symbol.strip():
        errors.append('symbol must be non-empty string')
    if payload.get('asset_class') not in {'stock', 'metal', 'forex'}:
        errors.append('asset_class must be one of stock|metal|forex')
    if payload.get('heuristic_signal_consensus') not in {'LONG', 'SHORT', 'NEUTRAL'}:
        errors.append('heuristic_signal_consensus must be LONG|SHORT|NEUTRAL')
    if not _is_iso_utc(payload.get('timestamp')):
        errors.append('timestamp must be ISO 8601 in UTC')
    trigger_reason = payload.get('trigger_reason')
    if not isinstance(trigger_reason, str) or not trigger_reason.strip():
        errors.append('trigger_reason must be non-empty string')
    trigger_reasons = payload.get('trigger_reasons')
    if trigger_reasons is not None:
        if not isinstance(trigger_reasons, list):
            errors.append('trigger_reasons must be an array of non-empty strings')
        elif not trigger_reasons:
            errors.append('trigger_reasons must not be empty when provided')
        elif not all((isinstance(reason, str) and reason.strip() for reason in trigger_reasons)):
            errors.append('trigger_reasons must contain non-empty strings')
        elif isinstance(trigger_reason, str) and trigger_reasons[0] != trigger_reason:
            errors.append('trigger_reason must match trigger_reasons[0]')
    composite_score = payload.get('composite_score')
    if not _is_finite_number(composite_score):
        errors.append('composite_score must be finite numeric')
    else:
        numeric_score = float(composite_score)
        if numeric_score < 0.0 or numeric_score > 1.0:
            errors.append('composite_score must be in [0.0, 1.0]')
    features = payload.get('features')
    if not isinstance(features, dict):
        errors.append('features must be an object')
    else:
        if 'current_price' not in features:
            errors.append('features.current_price is required')
        elif not _is_finite_number(features.get('current_price')):
            errors.append('features.current_price must be finite numeric')
        for key in ('rsi', 'macd_hist'):
            if key not in features:
                errors.append(f'features.{key} is required')
                continue
            value = features.get(key)
            if value is not None and (not _is_finite_number(value)):
                errors.append(f'features.{key} must be finite numeric or null')
    if not isinstance(payload.get('indicators'), dict):
        errors.append('indicators must be an object')
    else:
        _validate_indicators(payload['indicators'], errors)
    if not isinstance(payload.get('patterns'), dict):
        errors.append('patterns must be an object')
    else:
        _validate_patterns(payload['patterns'], errors)
    if not isinstance(payload.get('historical_snapshots'), list):
        errors.append('historical_snapshots must be an array')
    else:
        _validate_historical_snapshots(payload['historical_snapshots'], errors)
    entry_price_suggestion = payload.get('entry_price_suggestion')
    if not isinstance(entry_price_suggestion, str) or not entry_price_suggestion.strip():
        errors.append('entry_price_suggestion must be non-empty string')
    signal_log_db_id = payload.get('signal_log_db_id')
    if require_db_id:
        if not _is_int(signal_log_db_id):
            errors.append('signal_log_db_id must be integer before publish')
    elif signal_log_db_id is not None and (not _is_int(signal_log_db_id)):
        errors.append('signal_log_db_id must be null or integer')
    return errors

def validate_live_payload(payload: dict[str, Any], now_utc: dt.datetime | None=None, max_age_ms: int | None=None) -> list[str]:
    errors: list[str] = []
    required = ['symbol', 'asset_class', 'price', 'bid', 'ask', 'timestamp', 'ts']
    for key in required:
        if key not in payload:
            errors.append(f'missing field: {key}')
    symbol = payload.get('symbol')
    if not isinstance(symbol, str) or not symbol.strip():
        errors.append('symbol must be non-empty string')
    if payload.get('asset_class') not in {'stock', 'metal', 'forex'}:
        errors.append('asset_class must be one of stock|metal|forex')
    if not _is_finite_number(payload.get('price')):
        errors.append('price must be finite numeric')
    for key in ('bid', 'ask'):
        value = payload.get(key)
        if value is not None and (not _is_finite_number(value)):
            errors.append(f'{key} must be finite numeric or null')
    if not _is_iso_utc(payload.get('timestamp')):
        errors.append('timestamp must be ISO 8601 in UTC')
    ts = payload.get('ts')
    if not isinstance(ts, int):
        errors.append('ts must be integer unix ms timestamp')
    elif now_utc is not None and max_age_ms is not None:
        age_ms = int(now_utc.timestamp() * 1000) - ts
        if age_ms > int(max_age_ms):
            errors.append(f'ts is stale (age_ms={age_ms}, max_age_ms={int(max_age_ms)})')
    return errors

def _is_uuid_v4(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return parsed.version == 4 and str(parsed) == value.lower()

def _is_iso_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith('Z'):
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and (not isinstance(value, bool))

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and (not isinstance(value, bool))

def _is_finite_number(value: Any) -> bool:
    if not _is_number(value):
        return False
    return math.isfinite(float(value))

def _is_score(value: Any) -> bool:
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0

def _validate_indicators(indicators: dict[str, Any], errors: list[str]) -> None:
    if indicators.get('consensus') not in {'BULLISH', 'BEARISH', 'NEUTRAL'}:
        errors.append('indicators.consensus must be BULLISH|BEARISH|NEUTRAL')
    if not _is_score(indicators.get('consensus_strength')):
        errors.append('indicators.consensus_strength must be finite numeric in [0.0, 1.0]')
    signals = indicators.get('signals')
    if not isinstance(signals, list):
        errors.append('indicators.signals must be an array')
        return
    for idx, signal in enumerate(signals):
        prefix = f'indicators.signals[{idx}]'
        if not isinstance(signal, dict):
            errors.append(f'{prefix} must be an object')
            continue
        indicator_name = signal.get('indicator')
        if not isinstance(indicator_name, str) or not indicator_name.strip():
            errors.append(f'{prefix}.indicator must be non-empty string')
        if signal.get('signal') not in {'BULLISH', 'BEARISH', 'NEUTRAL'}:
            errors.append(f'{prefix}.signal must be BULLISH|BEARISH|NEUTRAL')
        if not _is_score(signal.get('strength')):
            errors.append(f'{prefix}.strength must be finite numeric in [0.0, 1.0]')

def _validate_patterns(patterns: dict[str, Any], errors: list[str]) -> None:
    if patterns.get('consensus') not in {'BULLISH', 'BEARISH', 'NEUTRAL'}:
        errors.append('patterns.consensus must be BULLISH|BEARISH|NEUTRAL')
    if not _is_score(patterns.get('consensus_strength')):
        errors.append('patterns.consensus_strength must be finite numeric in [0.0, 1.0]')
    info = patterns.get('detected_patterns_info')
    if not isinstance(info, list):
        errors.append('patterns.detected_patterns_info must be an array')
        return
    for idx, pattern in enumerate(info):
        prefix = f'patterns.detected_patterns_info[{idx}]'
        if not isinstance(pattern, dict):
            errors.append(f'{prefix} must be an object')
            continue
        pattern_name = pattern.get('pattern_name')
        if not isinstance(pattern_name, str) or not pattern_name.strip():
            errors.append(f'{prefix}.pattern_name must be non-empty string')
        if pattern.get('signal') not in {'BULLISH', 'BEARISH', 'NEUTRAL'}:
            errors.append(f'{prefix}.signal must be BULLISH|BEARISH|NEUTRAL')
        if not _is_score(pattern.get('strength')):
            errors.append(f'{prefix}.strength must be finite numeric in [0.0, 1.0]')
        if not _is_int(pattern.get('candle_offset')):
            errors.append(f'{prefix}.candle_offset must be integer')

def _validate_historical_snapshots(snapshots: list[Any], errors: list[str]) -> None:
    for idx, snapshot in enumerate(snapshots):
        prefix = f'historical_snapshots[{idx}]'
        if not isinstance(snapshot, dict):
            errors.append(f'{prefix} must be an object')
            continue
        if not _is_iso_utc(snapshot.get('timestamp')):
            errors.append(f'{prefix}.timestamp must be ISO 8601 in UTC')
        for key in ('close', 'volume', 'rsi', 'vol_rel', 'macd_hist'):
            value = snapshot.get(key)
            if value is not None and (not _is_finite_number(value)):
                errors.append(f'{prefix}.{key} must be finite numeric or null')
