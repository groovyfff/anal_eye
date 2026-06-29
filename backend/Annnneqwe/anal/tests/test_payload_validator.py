import datetime as dt
import uuid
from src.logic.payload_validator import validate_candidate_payload, validate_live_payload

def _candidate_payload() -> dict:
    return {'signal_id': str(uuid.uuid4()), 'symbol': 'AAPL', 'asset_class': 'stock', 'timestamp': dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).isoformat().replace('+00:00', 'Z'), 'trigger_reason': 'EMA_CROSSOVER_BULLISH', 'heuristic_signal_consensus': 'LONG', 'features': {'current_price': 123.4, 'rsi': 55.0, 'macd_hist': 0.2}, 'indicators': {'consensus': 'BULLISH', 'consensus_strength': 0.6, 'signals': []}, 'patterns': {'consensus': 'NEUTRAL', 'consensus_strength': 0.0, 'detected_patterns_info': []}, 'historical_snapshots': [], 'composite_score': 0.5, 'entry_price_suggestion': 'market', 'signal_log_db_id': None}

def _live_payload() -> dict:
    return {'symbol': 'AAPL', 'asset_class': 'stock', 'price': 123.4, 'bid': 123.3, 'ask': 123.5, 'timestamp': dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).isoformat().replace('+00:00', 'Z'), 'ts': 1767225600000}

def test_validate_candidate_payload_ok() -> None:
    errors = validate_candidate_payload(_candidate_payload())
    assert errors == []

def test_validate_candidate_payload_rejects_invalid_fields() -> None:
    payload = _candidate_payload()
    payload['signal_id'] = 'not-uuid'
    payload['asset_class'] = 'crypto'
    payload['composite_score'] = 5.0
    errors = validate_candidate_payload(payload)
    assert errors
    assert any(('UUID v4' in item for item in errors))
    assert any(('asset_class' in item for item in errors))
    assert any(('composite_score' in item for item in errors))

def test_validate_candidate_payload_rejects_non_finite_required_features() -> None:
    payload = _candidate_payload()
    payload['features']['current_price'] = float('nan')
    payload['features']['rsi'] = None
    payload['features']['macd_hist'] = float('inf')
    errors = validate_candidate_payload(payload)
    assert any(('features.current_price must be finite numeric' in item for item in errors))
    assert not any(('features.rsi' in item for item in errors))
    assert any(('features.macd_hist must be finite numeric or null' in item for item in errors))

def test_validate_candidate_payload_accepts_trigger_reasons_when_consistent() -> None:
    payload = _candidate_payload()
    payload['trigger_reasons'] = ['EMA_CROSSOVER_BULLISH', 'VOLUME_SPIKE']
    errors = validate_candidate_payload(payload)
    assert errors == []

def test_validate_candidate_payload_rejects_trigger_reasons_mismatch() -> None:
    payload = _candidate_payload()
    payload['trigger_reasons'] = ['VOLUME_SPIKE', 'EMA_CROSSOVER_BULLISH']
    errors = validate_candidate_payload(payload)
    assert any(('trigger_reason must match trigger_reasons[0]' in item for item in errors))

def test_validate_live_payload_ok() -> None:
    errors = validate_live_payload(_live_payload())
    assert errors == []

def test_validate_live_payload_rejects_non_int_ts() -> None:
    payload = _live_payload()
    payload['ts'] = '1767225600000'
    errors = validate_live_payload(payload)
    assert any(('ts must be integer' in item for item in errors))

def test_validate_live_payload_rejects_empty_symbol() -> None:
    payload = _live_payload()
    payload['symbol'] = '   '
    errors = validate_live_payload(payload)
    assert any(('symbol must be non-empty string' in item for item in errors))

def test_validate_live_payload_rejects_stale_ts_when_freshness_enabled() -> None:
    payload = _live_payload()
    now_utc = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    payload['ts'] = int(now_utc.timestamp() * 1000) - 5000
    errors = validate_live_payload(payload, now_utc=now_utc, max_age_ms=4500)
    assert any(('ts is stale' in item for item in errors))

def test_validate_live_payload_accepts_fresh_ts_when_freshness_enabled() -> None:
    payload = _live_payload()
    now_utc = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    payload['ts'] = int(now_utc.timestamp() * 1000) - 1500
    errors = validate_live_payload(payload, now_utc=now_utc, max_age_ms=4500)
    assert errors == []
