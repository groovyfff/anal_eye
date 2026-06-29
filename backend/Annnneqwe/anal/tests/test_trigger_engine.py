import pandas as pd
from src.logic.trigger_engine import TriggerEngine

def _base_frame() -> pd.DataFrame:
    return pd.DataFrame([{'close': 100.0, 'ema_8': 99.0, 'ema_21': 100.0, 'rsi': 25.0, 'vol_rel': 1.0, 'macd_hist': -0.5, 'support_nearest': 95.0, 'resistance_nearest': 105.0}, {'close': 101.0, 'ema_8': 101.5, 'ema_21': 100.2, 'rsi': 35.0, 'vol_rel': 1.2, 'macd_hist': 0.2, 'support_nearest': 95.0, 'resistance_nearest': 105.0}])

def _default_config() -> dict:
    return {'ema_crossover': {'enabled': True, 'fast_period': 8, 'slow_period': 21}, 'rsi_extreme': {'enabled': True, 'oversold_exit': 30, 'overbought_exit': 70}, 'volume_spike': {'enabled': True, 'threshold': 2.0}, 'macd_signal_cross': {'enabled': True}, 'price_breakout': {'enabled': True}}

def test_ema_crossover_trigger() -> None:
    cfg = _default_config()
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    result = TriggerEngine(cfg).evaluate(_base_frame())
    assert result.triggered
    assert 'EMA_CROSSOVER_BULLISH' in result.reasons

def test_rsi_extreme_trigger() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = _base_frame()
    frame.loc[1, 'rsi'] = 31.0
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert 'RSI_OVERSOLD_EXIT' in result.reasons

def test_volume_spike_trigger() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['rsi_extreme']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = _base_frame()
    frame.loc[1, 'vol_rel'] = 2.5
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert 'VOLUME_SPIKE' in result.reasons

def test_macd_cross_trigger() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = _base_frame()
    frame.loc[0, 'macd_hist'] = -0.1
    frame.loc[1, 'macd_hist'] = 0.1
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert 'MACD_BULLISH_CROSS' in result.reasons

def test_price_breakout_trigger() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    frame = _base_frame()
    frame.loc[1, 'close'] = 106.0
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert 'PRICE_BREAKOUT_RESISTANCE' in result.reasons

def test_volume_spike_nan_macd_uses_ema_direction() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['rsi_extreme']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = _base_frame()
    frame.loc[1, 'vol_rel'] = 3.0
    frame.loc[1, 'macd_hist'] = float('nan')
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert 'VOLUME_SPIKE' in result.reasons
    assert result.heuristic_signal_consensus == 'LONG'

def test_context_trend_participates_in_consensus() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['enabled'] = False
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = True
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = _base_frame()
    frame.loc[1, 'vol_rel'] = 2.5
    frame.loc[1, 'macd_hist'] = 0.1
    result = TriggerEngine(cfg).evaluate(frame, context_trends={'1h': {'direction': 'SHORT', 'strength': 0.9}, '4h': {'direction': 'SHORT', 'strength': 0.8}})
    assert result.triggered
    assert 'VOLUME_SPIKE' in result.reasons
    assert result.heuristic_signal_consensus == 'SHORT'
    assert any((item['indicator'] == 'context_trend_1h' for item in result.indicators['signals']))

def test_ema_crossover_uses_configured_periods() -> None:
    cfg = _default_config()
    cfg['ema_crossover']['fast_period'] = 5
    cfg['ema_crossover']['slow_period'] = 13
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = pd.DataFrame([{'ema_5': 10.0, 'ema_13': 11.0, 'ema_8': 12.0, 'ema_21': 11.0}, {'ema_5': 12.5, 'ema_13': 11.5, 'ema_8': 10.0, 'ema_21': 11.0}])
    result = TriggerEngine(cfg).evaluate(frame)
    assert result.triggered
    assert result.reasons == ['EMA_CROSSOVER_BULLISH']

def test_indicator_summary_ema_equal_is_neutral() -> None:
    cfg = _default_config()
    cfg['rsi_extreme']['enabled'] = False
    cfg['volume_spike']['enabled'] = False
    cfg['macd_signal_cross']['enabled'] = False
    cfg['price_breakout']['enabled'] = False
    frame = pd.DataFrame([{'close': 100.0, 'ema_8': 100.0, 'ema_21': 100.0}, {'close': 101.0, 'ema_8': 101.0, 'ema_21': 101.0}])
    result = TriggerEngine(cfg).evaluate(frame)
    ema_signal = next((item for item in result.indicators['signals'] if item['indicator'] == 'ema_crossover'))
    assert ema_signal['signal'] == 'NEUTRAL'
    assert ema_signal['strength'] == 0.0
