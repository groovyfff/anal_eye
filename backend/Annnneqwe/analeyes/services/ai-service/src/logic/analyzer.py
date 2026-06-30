from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INTERNAL_SOURCE_AI = 'ensemble'


class AssetClassAwareAnalyzer:
    """Internal rule-based analyzer (AssetClassAwareAnalyzer) — no external AI providers."""

    def __init__(self, prompts_dir: Path, min_composite: float = 0.5) -> None:
        self.prompts_dir = prompts_dir
        self.min_composite = min_composite

    def load_prompt_template(self, asset_class: str) -> str:
        asset_class = asset_class or 'crypto'
        for name in (f'{asset_class}_gpt_prompt.txt', 'stock_gpt_prompt.txt'):
            path = self.prompts_dir / name
            if path.is_file():
                return path.read_text(encoding='utf-8')
        return 'Internal analysis for {symbol} ({asset_class})'

    def _resolve_current_price(self, candidate: dict[str, Any]) -> float:
        features = candidate.get('features') or {}
        for value in (
            candidate.get('current_price'),
            features.get('current_price'),
            candidate.get('feat_current_price'),
            candidate.get('price'),
        ):
            if value is None:
                continue
            try:
                parsed = float(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return 0.0

    def _candle_close_trend(self, candles: list[dict[str, Any]]) -> float:
        closes: list[float] = []
        for candle in candles[-5:]:
            close = candle.get('close')
            if close is None:
                continue
            try:
                closes.append(float(close))
            except (TypeError, ValueError):
                continue
        if len(closes) < 2:
            return 0.0
        return closes[-1] - closes[0]

    def _infer_direction(
        self,
        candidate: dict[str, Any],
        *,
        direction_hint: str | None,
    ) -> tuple[str, float, list[str]]:
        """Internal scoring from market features and candles (project rule analyzer)."""
        features = candidate.get('features') or {}
        candles = candidate.get('candles') or candidate.get('historical_ohlcv') or []
        score = 0.0
        signals: list[str] = []

        ema_short = features.get('ema_short')
        ema_long = features.get('ema_long')
        if ema_short is not None and ema_long is not None:
            try:
                if float(ema_short) > float(ema_long):
                    score += 1.0
                    signals.append('ema_short>ema_long')
                elif float(ema_short) < float(ema_long):
                    score -= 1.0
                    signals.append('ema_short<ema_long')
            except (TypeError, ValueError):
                pass

        macd_hist = features.get('macd_hist')
        if macd_hist is not None:
            try:
                mh = float(macd_hist)
                if mh > 0:
                    score += 0.75
                    signals.append('macd_hist>0')
                elif mh < 0:
                    score -= 0.75
                    signals.append('macd_hist<0')
            except (TypeError, ValueError):
                pass

        rsi = features.get('rsi')
        if rsi is not None:
            try:
                r = float(rsi)
                if r < 35:
                    score += 0.5
                    signals.append('rsi_oversold')
                elif r > 65:
                    score -= 0.5
                    signals.append('rsi_overbought')
            except (TypeError, ValueError):
                pass

        price_change_1h = features.get('price_change_1h')
        if price_change_1h is not None:
            try:
                pc = float(price_change_1h)
                if pc > 0.5:
                    score += 0.35
                    signals.append('price_change_1h_positive')
                elif pc < -0.5:
                    score -= 0.35
                    signals.append('price_change_1h_negative')
            except (TypeError, ValueError):
                pass

        candle_delta = self._candle_close_trend(candles)
        if candle_delta > 0:
            score += 0.25
            signals.append('candle_close_rising')
        elif candle_delta < 0:
            score -= 0.25
            signals.append('candle_close_falling')

        market_state = str(candidate.get('market_state') or '').lower()
        if market_state == 'trend':
            if score > 0:
                score += 0.15
                signals.append('market_state_trend_align')
            elif score < 0:
                score -= 0.15

        if direction_hint == 'LONG':
            score += 0.15
            signals.append('weak_hint_long')
        elif direction_hint == 'SHORT':
            score -= 0.15
            signals.append('weak_hint_short')

        if score >= 1.0:
            return 'LONG', score, signals
        if score <= -1.0:
            return 'SHORT', score, signals
        return 'SKIP', score, signals

    def analyze(self, candidate: dict[str, Any], *, accept_manual_test: bool = False) -> dict[str, Any]:
        _ = accept_manual_test  # only used by main for DB persist gate
        asset_class = str(candidate.get('asset_class') or '')
        symbol = candidate.get('symbol')
        features = candidate.get('features')
        direction_hint = candidate.get('direction_hint')
        composite = float(candidate.get('composite_score') or 0.0)
        current_price = self._resolve_current_price(candidate)

        template = self.load_prompt_template(asset_class)
        prompt_preview = (
            template.replace('{symbol}', str(symbol or ''))
            .replace('{asset_class}', asset_class)
            .replace('{features}', json.dumps(features or {}, ensure_ascii=False)[:500])
        )

        skip_parts: list[str] = []
        if not symbol:
            skip_parts.append('missing_symbol')
        if not asset_class:
            skip_parts.append('missing_asset_class')
        if features is None or (isinstance(features, dict) and not features):
            skip_parts.append('missing_features')
        if composite < self.min_composite:
            skip_parts.append(f'composite_below_threshold: {composite} < {self.min_composite}')
        if current_price <= 0:
            skip_parts.append(f'missing_or_invalid_current_price: {current_price!r}')

        if skip_parts:
            skip_reason = '; '.join(skip_parts)
            return {
                'decision': 'SKIP',
                'skip_reason': skip_reason,
                'confidence': 0.0,
                'reason': skip_reason,
                'source_ai': INTERNAL_SOURCE_AI,
                'prompt_preview': prompt_preview,
            }

        logger.info(
            'Running internal AI analysis symbol=%s direction_hint=%s',
            symbol,
            direction_hint,
        )

        decision, feature_score, signals = self._infer_direction(candidate, direction_hint=direction_hint)

        if decision == 'SKIP':
            skip_reason = f'internal_analyzer_no_clear_edge: score={feature_score} signals={signals}'
            logger.info(
                'Internal AI final decision=SKIP source_ai=%s reason=%s',
                INTERNAL_SOURCE_AI,
                skip_reason,
            )
            return {
                'decision': 'SKIP',
                'skip_reason': skip_reason,
                'confidence': round(min(0.99, max(composite, 0.3)), 4),
                'reason': skip_reason,
                'source_ai': INTERNAL_SOURCE_AI,
                'feature_score': feature_score,
                'signals': signals,
                'prompt_preview': prompt_preview,
            }

        if decision == 'LONG':
            tp = round(current_price * 1.02, 4)
            sl = round(current_price * 0.99, 4)
        else:
            tp = round(current_price * 0.98, 4)
            sl = round(current_price * 1.01, 4)

        reason = f'Internal rule analyzer: score={feature_score} signals={signals}'
        if direction_hint:
            reason += f' (weak direction_hint={direction_hint})'

        logger.info(
            'Internal AI final decision=%s source_ai=%s reason=%s',
            decision,
            INTERNAL_SOURCE_AI,
            reason,
        )

        return {
            'decision': decision,
            'signal_type': decision,
            'confidence': round(min(0.99, max(composite, 0.5)), 4),
            'reason': reason,
            'reason_summary': reason,
            'entry_price': 'market',
            'tp': tp,
            'tp_price': tp,
            'sl': sl,
            'sl_price': sl,
            'leverage': 1.0,
            'consensus_achieved': True,
            'source_ai': INTERNAL_SOURCE_AI,
            'feature_score': feature_score,
            'signals': signals,
            'prompt_preview': prompt_preview,
            'manual_test': not candidate.get('signal_log_db_id'),
        }
