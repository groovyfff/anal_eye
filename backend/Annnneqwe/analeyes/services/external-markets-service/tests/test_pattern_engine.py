import pandas as pd
from src.logic.pattern_engine import PatternEngine

def _frame_with_bullish_engulfing() -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=3, freq='5min', tz='UTC')
    return pd.DataFrame({'open': [10.0, 10.2, 8.9], 'high': [10.3, 10.3, 10.7], 'low': [9.7, 8.7, 8.8], 'close': [10.1, 9.0, 10.6]}, index=index)

def test_pattern_engine_disabled_returns_empty_payload() -> None:
    engine = PatternEngine({'enabled': False})
    result = engine.analyze(_frame_with_bullish_engulfing())
    assert result.pattern_score == 0.0
    assert result.patterns_payload['consensus'] == 'NEUTRAL'
    assert result.patterns_payload['detected_patterns_info'] == []

def test_pattern_engine_detects_directional_patterns() -> None:
    engine = PatternEngine({'enabled': True, 'lookback_candles': 6, 'max_detected': 6, 'min_strength': 0.2})
    result = engine.analyze(_frame_with_bullish_engulfing())
    assert 0.0 <= result.pattern_score <= 1.0
    assert result.patterns_payload['consensus'] in {'BULLISH', 'BEARISH', 'NEUTRAL'}
    assert isinstance(result.patterns_payload['detected_patterns_info'], list)
    assert result.patterns_payload['detected_patterns_info']
    assert any((item['signal'] in {'BULLISH', 'BEARISH'} for item in result.patterns_payload['detected_patterns_info']))

def test_pattern_engine_graceful_when_required_columns_missing() -> None:
    frame = pd.DataFrame({'close': [1.0, 2.0]}, index=pd.date_range('2026-01-01', periods=2, freq='5min', tz='UTC'))
    engine = PatternEngine({'enabled': True})
    result = engine.analyze(frame)
    assert result.pattern_score == 0.0
    assert result.patterns_payload['detected_patterns_info'] == []
