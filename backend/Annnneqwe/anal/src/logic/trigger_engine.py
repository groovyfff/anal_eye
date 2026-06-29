from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass(slots=True)
class TriggerResult:
    triggered: bool
    reasons: list[str]
    heuristic_signal_consensus: str
    indicators: dict[str, Any]

class TriggerEngine:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        ema_cfg = self.config.get('ema_crossover', {})
        fast_period = max(1, int(ema_cfg.get('fast_period', 8)))
        slow_period = max(2, int(ema_cfg.get('slow_period', 21)))
        if slow_period <= fast_period:
            slow_period = fast_period + 1
        self.ema_fast_period = fast_period
        self.ema_slow_period = slow_period

    def evaluate(self, indicator_df: pd.DataFrame, context_trends: dict[str, dict[str, Any]] | None=None) -> TriggerResult:
        context = context_trends or {}
        if len(indicator_df) < 2:
            return TriggerResult(triggered=False, reasons=[], heuristic_signal_consensus='NEUTRAL', indicators=self._build_indicators_summary(indicator_df, context))
        prev = indicator_df.iloc[-2]
        curr = indicator_df.iloc[-1]
        reasons: list[str] = []
        directions: list[str] = []
        if self.config.get('ema_crossover', {}).get('enabled', True):
            reason = self._ema_crossover(prev, curr)
            if reason:
                reasons.append(reason)
                directions.append('LONG' if 'BULLISH' in reason else 'SHORT')
        if self.config.get('rsi_extreme', {}).get('enabled', True):
            reason = self._rsi_extreme(prev, curr)
            if reason:
                reasons.append(reason)
                directions.append('LONG' if 'OVERSOLD' in reason else 'SHORT')
        if self.config.get('volume_spike', {}).get('enabled', True):
            reason, direction = self._volume_spike(curr)
            if reason:
                reasons.append(reason)
            if direction:
                directions.append(direction)
        if self.config.get('macd_signal_cross', {}).get('enabled', True):
            reason = self._macd_cross(prev, curr)
            if reason:
                reasons.append(reason)
                directions.append('LONG' if 'BULLISH' in reason else 'SHORT')
        if self.config.get('price_breakout', {}).get('enabled', True):
            reason = self._price_breakout(curr)
            if reason:
                reasons.append(reason)
                directions.append('LONG' if 'RESISTANCE' in reason else 'SHORT')
        directions.extend(self._context_directions(context))
        consensus = self._consensus_from_directions(directions)
        return TriggerResult(triggered=bool(reasons), reasons=reasons, heuristic_signal_consensus=consensus, indicators=self._build_indicators_summary(indicator_df, context))

    def _ema_crossover(self, prev: pd.Series, curr: pd.Series) -> str | None:
        prev_fast = self._ema_fast(prev)
        prev_slow = self._ema_slow(prev)
        curr_fast = self._ema_fast(curr)
        curr_slow = self._ema_slow(curr)
        if any((value is None for value in (prev_fast, prev_slow, curr_fast, curr_slow))):
            return None
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return 'EMA_CROSSOVER_BULLISH'
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return 'EMA_CROSSOVER_BEARISH'
        return None

    def _rsi_extreme(self, prev: pd.Series, curr: pd.Series) -> str | None:
        cfg = self.config.get('rsi_extreme', {})
        oversold_exit = float(cfg.get('oversold_exit', 30))
        overbought_exit = float(cfg.get('overbought_exit', 70))
        prev_rsi = prev.get('rsi')
        curr_rsi = curr.get('rsi')
        if pd.isna([prev_rsi, curr_rsi]).any():
            return None
        if prev_rsi < oversold_exit and curr_rsi > oversold_exit:
            return 'RSI_OVERSOLD_EXIT'
        if prev_rsi > overbought_exit and curr_rsi < overbought_exit:
            return 'RSI_OVERBOUGHT_EXIT'
        return None

    def _volume_spike(self, curr: pd.Series) -> tuple[str | None, str | None]:
        cfg = self.config.get('volume_spike', {})
        threshold = float(cfg.get('threshold', 2.0))
        vol_rel = curr.get('vol_rel')
        if pd.isna(vol_rel) or float(vol_rel) <= threshold:
            return (None, None)
        return ('VOLUME_SPIKE', self._volume_spike_direction(curr))

    @staticmethod
    def _macd_cross(prev: pd.Series, curr: pd.Series) -> str | None:
        prev_hist = prev.get('macd_hist')
        curr_hist = curr.get('macd_hist')
        if pd.isna([prev_hist, curr_hist]).any():
            return None
        if prev_hist <= 0 and curr_hist > 0:
            return 'MACD_BULLISH_CROSS'
        if prev_hist >= 0 and curr_hist < 0:
            return 'MACD_BEARISH_CROSS'
        return None

    def _price_breakout(self, curr: pd.Series) -> str | None:
        close = curr.get('close')
        support = curr.get('support_nearest')
        resistance = curr.get('resistance_nearest')
        if pd.isna([close, support, resistance]).any():
            return None
        if close > resistance:
            return 'PRICE_BREAKOUT_RESISTANCE'
        if close < support:
            return 'PRICE_BREAKDOWN_SUPPORT'
        return None

    def _build_indicators_summary(self, indicator_df: pd.DataFrame, context_trends: dict[str, dict[str, Any]] | None=None) -> dict[str, Any]:
        context = context_trends or {}
        if indicator_df.empty:
            return {'consensus': 'NEUTRAL', 'consensus_strength': 0.0, 'signals': []}
        row = indicator_df.iloc[-1]
        signals: list[dict[str, Any]] = []
        bull = 0.0
        bear = 0.0
        ema_fast = self._ema_fast(row)
        ema_slow = self._ema_slow(row)
        close = row.get('close')
        if ema_fast is not None and ema_slow is not None and pd.notna(close):
            if ema_fast > ema_slow:
                direction = 'BULLISH'
                strength = min(abs(ema_fast - ema_slow) / max(float(close), 1e-09) * 25.0, 1.0)
            elif ema_fast < ema_slow:
                direction = 'BEARISH'
                strength = min(abs(ema_fast - ema_slow) / max(float(close), 1e-09) * 25.0, 1.0)
            else:
                direction = 'NEUTRAL'
                strength = 0.0
            signals.append({'indicator': 'ema_crossover', 'signal': direction, 'strength': round(strength, 4)})
            if direction == 'BULLISH':
                bull += strength
            elif direction == 'BEARISH':
                bear += strength
        macd_hist = row.get('macd_hist')
        if pd.notna(macd_hist):
            strength = min(abs(float(macd_hist)) / 2.0, 1.0)
            direction = 'BULLISH' if macd_hist > 0 else 'BEARISH' if macd_hist < 0 else 'NEUTRAL'
            signals.append({'indicator': 'macd_signals', 'signal': direction, 'strength': round(strength, 4)})
            if direction == 'BULLISH':
                bull += strength
            elif direction == 'BEARISH':
                bear += strength
        rsi = row.get('rsi')
        if pd.notna(rsi):
            if rsi < 35:
                direction = 'BULLISH'
                strength = min((35 - float(rsi)) / 35.0, 1.0)
                bull += strength
            elif rsi > 65:
                direction = 'BEARISH'
                strength = min((float(rsi) - 65) / 35.0, 1.0)
                bear += strength
            else:
                direction = 'NEUTRAL'
                strength = 0.4
            signals.append({'indicator': 'rsi_conditions', 'signal': direction, 'strength': round(strength, 4)})
        for timeframe in sorted(context.keys(), key=self._timeframe_sort_key):
            trend = context.get(timeframe) or {}
            direction = str(trend.get('direction', 'NEUTRAL')).upper()
            strength = self._normalized_strength(trend.get('strength'))
            signal = 'NEUTRAL'
            if direction == 'LONG':
                signal = 'BULLISH'
                bull += strength * 0.5
            elif direction == 'SHORT':
                signal = 'BEARISH'
                bear += strength * 0.5
            signals.append({'indicator': f'context_trend_{timeframe}', 'signal': signal, 'strength': round(strength, 4)})
        consensus = 'NEUTRAL'
        if bull > bear:
            consensus = 'BULLISH'
        elif bear > bull:
            consensus = 'BEARISH'
        total = bull + bear
        consensus_strength = 0.0 if total == 0 else abs(bull - bear) / total
        return {'consensus': consensus, 'consensus_strength': round(consensus_strength, 4), 'signals': signals}

    @staticmethod
    def _consensus_from_directions(directions: list[str]) -> str:
        long_votes = sum((1 for item in directions if item == 'LONG'))
        short_votes = sum((1 for item in directions if item == 'SHORT'))
        if long_votes > short_votes:
            return 'LONG'
        if short_votes > long_votes:
            return 'SHORT'
        return 'NEUTRAL'

    def _ema_fast(self, row: pd.Series) -> float | None:
        candidates = ('ema_fast', f'ema_{self.ema_fast_period}', 'ema_8')
        return self._first_numeric(row, candidates)

    def _ema_slow(self, row: pd.Series) -> float | None:
        candidates = ('ema_slow', f'ema_{self.ema_slow_period}', 'ema_21')
        return self._first_numeric(row, candidates)

    def _volume_spike_direction(self, curr: pd.Series) -> str | None:
        macd_hist = curr.get('macd_hist')
        if pd.notna(macd_hist):
            return 'LONG' if float(macd_hist) >= 0 else 'SHORT'
        ema_fast = self._ema_fast(curr)
        ema_slow = self._ema_slow(curr)
        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                return 'LONG'
            if ema_fast < ema_slow:
                return 'SHORT'
        close = curr.get('close')
        if ema_fast is not None and pd.notna(close):
            return 'LONG' if float(close) >= ema_fast else 'SHORT'
        return None

    @staticmethod
    def _first_numeric(row: pd.Series, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = row.get(key)
            if pd.notna(value):
                return float(value)
        return None

    def _context_directions(self, context_trends: dict[str, dict[str, Any]]) -> list[str]:
        directions: list[str] = []
        for timeframe in sorted(context_trends.keys(), key=self._timeframe_sort_key):
            trend = context_trends.get(timeframe) or {}
            direction = str(trend.get('direction', 'NEUTRAL')).upper()
            strength = self._normalized_strength(trend.get('strength'))
            if direction in {'LONG', 'SHORT'} and strength >= 0.15:
                directions.append(direction)
        return directions

    @staticmethod
    def _normalized_strength(value: Any) -> float:
        try:
            strength = float(value)
        except (TypeError, ValueError):
            return 0.0
        if pd.isna(strength):
            return 0.0
        return max(0.0, min(strength, 1.0))

    @staticmethod
    def _timeframe_sort_key(timeframe: str) -> tuple[int, int]:
        if timeframe.endswith('m') and timeframe[:-1].isdigit():
            return (0, int(timeframe[:-1]))
        if timeframe.endswith('h') and timeframe[:-1].isdigit():
            return (1, int(timeframe[:-1]))
        if timeframe.endswith('d') and timeframe[:-1].isdigit():
            return (2, int(timeframe[:-1]))
        return (3, 0)
