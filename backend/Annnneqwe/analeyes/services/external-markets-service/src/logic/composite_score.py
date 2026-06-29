from __future__ import annotations
from typing import Any

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

class CompositeScore:

    def __init__(self, config: dict[str, Any]) -> None:
        self.weights = config.get('weights', {})
        self.min_publish_score = float(config.get('min_publish_score', 0.0))

    def calculate(self, features: dict[str, Any], pattern_score: float=0.0) -> float:
        vol_rel = _as_float(features.get('vol_rel'), 0.0)
        rsi = _as_float(features.get('rsi'), 50.0)
        macd_hist = _as_float(features.get('macd_hist'), 0.0)
        adx = _as_float(features.get('adx'), 0.0)
        vol_rel_score = _clamp01(vol_rel / 3.0)
        rsi_score = _clamp01(abs(rsi - 50.0) / 50.0)
        macd_hist_score = _clamp01(abs(macd_hist) / 2.0)
        adx_score = _clamp01(adx / 50.0)
        pattern_score = _clamp01(pattern_score)
        weighted = {'vol_rel': vol_rel_score, 'rsi_score': rsi_score, 'macd_hist_score': macd_hist_score, 'adx_score': adx_score, 'pattern_score': pattern_score}
        total_weight = 0.0
        total_score = 0.0
        for key, metric_score in weighted.items():
            w = float(self.weights.get(key, 0.0))
            total_weight += w
            total_score += w * metric_score
        if total_weight <= 0:
            return 0.0
        return _clamp01(total_score / total_weight)

    def should_publish(self, score: float) -> bool:
        return score >= self.min_publish_score
