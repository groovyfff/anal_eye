import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import src.logic.feature_generator as feature_generator_module
from src.logic.feature_generator import FeatureGenerator

def _make_frame(size: int=250) -> pd.DataFrame:
    idx = pd.date_range('2026-01-01', periods=size, freq='5min', tz='UTC')
    close = np.linspace(100.0, 130.0, size)
    frame = pd.DataFrame({'open': close - 0.2, 'high': close + 0.5, 'low': close - 0.5, 'close': close, 'volume': np.linspace(1000.0, 2000.0, size)}, index=idx)
    return frame

def test_compute_indicators_contains_rsi_macd_ema() -> None:
    generator = FeatureGenerator({'breakout_lookback': 20, 'volume_window': 20})
    frame = _make_frame()
    out = generator.compute_indicators(frame)
    assert 'rsi' in out.columns
    assert 'macd_line' in out.columns
    assert 'ema_8' in out.columns
    assert pd.notna(out['rsi'].iloc[-1])
    assert pd.notna(out['macd_line'].iloc[-1])
    assert pd.notna(out['ema_8'].iloc[-1])

def test_build_feature_payload_returns_null_for_insufficient_data() -> None:
    generator = FeatureGenerator({'breakout_lookback': 20, 'volume_window': 20})
    short_frame = _make_frame(size=5)
    out = generator.compute_indicators(short_frame)
    features = generator.build_feature_payload(out, asset_class='stock', bid=None, ask=None)
    assert features['rsi'] is None
    assert features['macd_hist'] is None
    assert features['ema_short'] is None

def test_relative_volume_uses_matching_session_slot_baseline() -> None:
    generator = FeatureGenerator({'breakout_lookback': 20, 'volume_window': 20})
    idx = pd.date_range('2026-01-01 14:30:00+00:00', periods=25, freq='1D')
    volume = np.arange(1, 26, dtype=float)
    frame = pd.DataFrame({'open': 100.0 + volume, 'high': 101.0 + volume, 'low': 99.0 + volume, 'close': 100.5 + volume, 'volume': volume}, index=idx)
    out = generator.compute_indicators(frame, asset_class='stock')
    assert pd.isna(out['vol_rel'].iloc[19])
    assert pd.notna(out['vol_rel'].iloc[20])
    assert out['vol_rel'].iloc[20] == pytest.approx(2.0, rel=1e-06)

def test_vwap_resets_on_forex_rollover() -> None:
    generator = FeatureGenerator({'breakout_lookback': 20, 'volume_window': 20})
    idx = pd.DatetimeIndex([pd.Timestamp('2026-01-05T21:55:00Z'), pd.Timestamp('2026-01-05T22:00:00Z')])
    frame = pd.DataFrame({'open': [100.0, 110.0], 'high': [100.0, 110.0], 'low': [100.0, 110.0], 'close': [100.0, 110.0], 'volume': [10.0, 10.0]}, index=idx)
    out = generator.compute_indicators(frame, asset_class='forex')
    assert out['vwap'].iloc[0] == pytest.approx(100.0, rel=1e-06)
    assert out['vwap'].iloc[1] == pytest.approx(110.0, rel=1e-06)

def test_ema_fast_and_slow_use_config_periods() -> None:
    generator = FeatureGenerator({'breakout_lookback': 20, 'volume_window': 20, 'ema_fast_period': 5, 'ema_slow_period': 13})
    frame = _make_frame(size=60)
    out = generator.compute_indicators(frame)
    assert 'ema_fast' in out.columns
    assert 'ema_slow' in out.columns
    assert 'ema_5' in out.columns
    assert 'ema_13' in out.columns
    assert pd.notna(out['ema_fast'].iloc[-1])
    assert pd.notna(out['ema_slow'].iloc[-1])

def test_historical_snapshots_include_latest_closed_candle() -> None:
    frame = pd.DataFrame({'close': [100.0, 101.0, 102.0], 'volume': [10.0, 11.0, 12.0], 'rsi': [40.0, 50.0, 60.0], 'vol_rel': [0.8, 1.0, 1.2], 'macd_hist': [-0.1, 0.0, 0.1]}, index=pd.DatetimeIndex([pd.Timestamp('2026-01-01T10:00:00Z'), pd.Timestamp('2026-01-01T10:05:00Z'), pd.Timestamp('2026-01-01T10:10:00Z')]))
    snapshots = FeatureGenerator.build_historical_snapshots(frame, count=2)
    assert [item['timestamp'] for item in snapshots] == ['2026-01-01T10:10:00Z', '2026-01-01T10:05:00Z']
    assert snapshots[0]['close'] == pytest.approx(102.0, rel=1e-09)
    assert snapshots[1]['close'] == pytest.approx(101.0, rel=1e-09)

def test_ema_formula_matches_fallback_ewm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_generator_module, 'ta', None)
    series = pd.Series(np.linspace(100.0, 120.0, 30))
    ema = FeatureGenerator._ema(series, length=8)
    expected = series.ewm(span=8, adjust=False, min_periods=8).mean()
    pdt.assert_series_equal(ema, expected, check_names=False)

def test_macd_formula_matches_fallback_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_generator_module, 'ta', None)
    series = pd.Series(np.linspace(100.0, 130.0, 80))
    macd_line, macd_signal, macd_hist = FeatureGenerator._macd(series)
    expected_line = series.ewm(span=12, adjust=False, min_periods=12).mean() - series.ewm(span=26, adjust=False, min_periods=26).mean()
    expected_signal = expected_line.ewm(span=9, adjust=False, min_periods=9).mean()
    expected_hist = expected_line - expected_signal
    pdt.assert_series_equal(macd_line, expected_line, check_names=False)
    pdt.assert_series_equal(macd_signal, expected_signal, check_names=False)
    pdt.assert_series_equal(macd_hist, expected_hist, check_names=False)
    mask = macd_line.notna() & macd_signal.notna()
    assert (macd_hist[mask] - (macd_line[mask] - macd_signal[mask])).abs().max() <= 1e-12

def test_rsi_formula_handles_trend_and_flat_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_generator_module, 'ta', None)
    length = 14
    rising = pd.Series(np.linspace(1.0, 40.0, 40))
    falling = pd.Series(np.linspace(40.0, 1.0, 40))
    flat = pd.Series(np.full(40, 10.0))
    rsi_rising = FeatureGenerator._rsi(rising, length=length)
    rsi_falling = FeatureGenerator._rsi(falling, length=length)
    rsi_flat = FeatureGenerator._rsi(flat, length=length)
    assert rsi_rising.iloc[-1] == pytest.approx(100.0, rel=1e-09)
    assert rsi_falling.iloc[-1] == pytest.approx(0.0, rel=1e-09)
    assert rsi_flat.iloc[-1] == pytest.approx(50.0, rel=1e-09)
