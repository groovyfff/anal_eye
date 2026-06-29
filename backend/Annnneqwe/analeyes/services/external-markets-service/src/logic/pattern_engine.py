from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
try:
    import talib
except Exception:
    talib = None

@dataclass(slots=True)
class PatternDetection:
    patterns_payload: dict[str, Any]
    pattern_score: float

class PatternEngine:

    def __init__(self, config: dict[str, Any] | None=None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get('enabled', True))
        self.lookback_candles = max(1, int(cfg.get('lookback_candles', 8)))
        self.max_detected = max(1, int(cfg.get('max_detected', 6)))
        self.min_strength = self._clamp01(float(cfg.get('min_strength', 0.35)))

    def analyze(self, frame: pd.DataFrame) -> PatternDetection:
        if not self.enabled:
            return PatternDetection(patterns_payload=self._empty_payload(), pattern_score=0.0)
        required = {'open', 'high', 'low', 'close'}
        if frame.empty or not required.issubset(frame.columns):
            return PatternDetection(patterns_payload=self._empty_payload(), pattern_score=0.0)
        work = frame[['open', 'high', 'low', 'close']].copy()
        for col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')
        work = work.dropna()
        if len(work) < 2:
            return PatternDetection(patterns_payload=self._empty_payload(), pattern_score=0.0)
        entries = self._talib_patterns(work)
        if not entries:
            entries = self._fallback_patterns(work)
        directional = [item for item in entries if item['signal'] in {'BULLISH', 'BEARISH'}]
        if not directional:
            return PatternDetection(patterns_payload=self._empty_payload(), pattern_score=0.0)
        directional.sort(key=lambda item: float(item['strength']), reverse=True)
        detected = directional[:self.max_detected]
        bull = sum((float(item['strength']) for item in detected if item['signal'] == 'BULLISH'))
        bear = sum((float(item['strength']) for item in detected if item['signal'] == 'BEARISH'))
        consensus = 'NEUTRAL'
        if bull > bear:
            consensus = 'BULLISH'
        elif bear > bull:
            consensus = 'BEARISH'
        total = bull + bear
        consensus_strength = 0.0 if total == 0 else abs(bull - bear) / total
        pattern_score = float(np.clip(np.mean([float(item['strength']) for item in detected]), 0.0, 1.0))
        return PatternDetection(patterns_payload={'consensus': consensus, 'consensus_strength': round(consensus_strength, 4), 'detected_patterns_info': detected}, pattern_score=round(pattern_score, 4))

    def _talib_patterns(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if talib is None:
            return []
        specs = [('Engulfing', talib.CDLENGULFING, 0.65), ('Morning Star', talib.CDLMORNINGSTAR, 0.9), ('Evening Star', talib.CDLEVENINGSTAR, 0.9), ('Hammer', talib.CDLHAMMER, 0.6), ('Hanging Man', talib.CDLHANGINGMAN, 0.6), ('Shooting Star', talib.CDLSHOOTINGSTAR, 0.6), ('Three White Soldiers', talib.CDL3WHITESOLDIERS, 0.95), ('Three Black Crows', talib.CDL3BLACKCROWS, 0.95), ('Piercing', talib.CDLPIERCING, 0.7), ('Dark Cloud Cover', talib.CDLDARKCLOUDCOVER, 0.7)]
        open_ = frame['open'].to_numpy(dtype='float64')
        high = frame['high'].to_numpy(dtype='float64')
        low = frame['low'].to_numpy(dtype='float64')
        close = frame['close'].to_numpy(dtype='float64')
        entries: list[dict[str, Any]] = []
        lookback_start = max(0, len(frame) - self.lookback_candles)
        for name, fn, base_strength in specs:
            try:
                values = fn(open_, high, low, close)
            except Exception:
                continue
            for idx in range(lookback_start, len(values)):
                raw = int(values[idx])
                if raw == 0:
                    continue
                signal = 'BULLISH' if raw > 0 else 'BEARISH'
                magnitude = min(abs(raw) / 100.0, 2.0)
                strength = self._clamp01(base_strength * min(magnitude, 1.5))
                if strength < self.min_strength:
                    continue
                entries.append({'pattern_name': name, 'signal': signal, 'strength': round(strength, 4), 'candle_offset': int(idx - (len(frame) - 1))})
        return entries

    def _fallback_patterns(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        open_s = frame['open']
        high_s = frame['high']
        low_s = frame['low']
        close_s = frame['close']
        body = (close_s - open_s).abs()
        range_ = (high_s - low_s).replace(0, np.nan)
        upper_shadow = high_s - np.maximum(open_s, close_s)
        lower_shadow = np.minimum(open_s, close_s) - low_s
        lookback_start = max(1, len(frame) - self.lookback_candles)
        for idx in range(lookback_start, len(frame)):
            prev_idx = idx - 1
            o = float(open_s.iloc[idx])
            c = float(close_s.iloc[idx])
            po = float(open_s.iloc[prev_idx])
            pc = float(close_s.iloc[prev_idx])
            body_ratio = self._safe_ratio(float(body.iloc[idx]), float(range_.iloc[idx]))
            upper_ratio = self._safe_ratio(float(upper_shadow.iloc[idx]), float(range_.iloc[idx]))
            lower_ratio = self._safe_ratio(float(lower_shadow.iloc[idx]), float(range_.iloc[idx]))
            if pc < po and c > o and (c >= po) and (o <= pc):
                entries.append(self._entry('Engulfing', 'BULLISH', 0.7, idx, len(frame)))
            if pc > po and c < o and (o >= pc) and (c <= po):
                entries.append(self._entry('Engulfing', 'BEARISH', 0.7, idx, len(frame)))
            if body_ratio <= 0.35 and lower_ratio >= 0.5 and (upper_ratio <= 0.2):
                entries.append(self._entry('Hammer', 'BULLISH', 0.55, idx, len(frame)))
            if body_ratio <= 0.35 and upper_ratio >= 0.5 and (lower_ratio <= 0.2):
                entries.append(self._entry('Shooting Star', 'BEARISH', 0.55, idx, len(frame)))
            if body_ratio <= 0.08:
                direction = 'BULLISH' if pc < po else 'BEARISH' if pc > po else 'NEUTRAL'
                if direction != 'NEUTRAL':
                    entries.append(self._entry('Doji', direction, 0.4, idx, len(frame)))
            if idx >= 2:
                o2 = float(open_s.iloc[idx - 2])
                c2 = float(close_s.iloc[idx - 2])
                mid = (o2 + c2) / 2.0
                small_middle = self._safe_ratio(float(body.iloc[idx - 1]), float(range_.iloc[idx - 1])) <= 0.35
                if c2 < o2 and small_middle and (c > o) and (c >= mid):
                    entries.append(self._entry('Morning Star', 'BULLISH', 0.8, idx, len(frame)))
                if c2 > o2 and small_middle and (c < o) and (c <= mid):
                    entries.append(self._entry('Evening Star', 'BEARISH', 0.8, idx, len(frame)))
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for item in sorted(entries, key=lambda x: (x['candle_offset'], x['pattern_name'], x['signal'])):
            key = (str(item['pattern_name']), str(item['signal']), int(item['candle_offset']))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return [item for item in deduped if float(item['strength']) >= self.min_strength]

    def _entry(self, name: str, signal: str, strength: float, idx: int, length: int) -> dict[str, Any]:
        return {'pattern_name': name, 'signal': signal, 'strength': round(self._clamp01(strength), 4), 'candle_offset': int(idx - (length - 1))}

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0 or np.isnan(denominator):
            return 0.0
        return float(np.clip(numerator / denominator, 0.0, 10.0))

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {'consensus': 'NEUTRAL', 'consensus_strength': 0.0, 'detected_patterns_info': []}
