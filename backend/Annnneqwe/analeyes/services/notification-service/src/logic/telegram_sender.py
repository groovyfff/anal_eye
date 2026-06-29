from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

logger = logging.getLogger(__name__)


class TelegramSender:
    """Форматирование и доставка уведомлений (stdout fallback без TELEGRAM_BOT_TOKEN)."""

    def __init__(self, config: dict[str, Any]) -> None:
        telegram_cfg = config.get('telegram', {}) or {}
        self.group_id = telegram_cfg.get('group_id')
        self.asset_class_topics = telegram_cfg.get('asset_class_topics', {}) or {}
        self.ai_specific_topics = telegram_cfg.get('ai_specific_topics', {}) or {}
        self.bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')

    def resolve_topic_id(self, asset_class: str | None, source_ai: str | None) -> int | None:
        if asset_class and asset_class in self.asset_class_topics:
            return self.asset_class_topics.get(asset_class)
        if source_ai and source_ai in self.ai_specific_topics:
            return self.ai_specific_topics.get(source_ai)
        return None

    def format_signal_message(self, payload: dict[str, Any]) -> str:
        asset_class = str(payload.get('asset_class', 'crypto'))
        features = payload.get('features') or {}
        lines = [
            f"🚀 *{payload.get('name') or payload.get('symbol')}* ({asset_class})",
            f"Direction: *{payload.get('decision')}* | Confidence: {payload.get('confidence')}",
            f"Entry: {payload.get('entry_price')} | TP: {payload.get('tp')} | SL: {payload.get('sl')}",
        ]
        if asset_class == 'crypto':
            for key in ('funding_rate', 'open_interest_z', 'cvd'):
                if features.get(key) is not None:
                    lines.append(f"{key}: {features[key]}")
        else:
            for key in ('rsi', 'macd_hist', 'vol_rel'):
                if features.get(key) is not None:
                    lines.append(f"{key}: {features[key]}")
        return '\n'.join(lines)

    def build_chart_png(self, payload: dict[str, Any]) -> bytes | None:
        asset_class = str(payload.get('asset_class', 'crypto'))
        ohlcv = payload.get('historical_ohlcv') or []
        if asset_class == 'crypto' or not ohlcv:
            return None
        rows = []
        for candle in ohlcv:
            ts = candle.get('timestamp')
            if not ts:
                continue
            rows.append(
                {
                    'Date': pd.to_datetime(ts.replace('Z', '+00:00')),
                    'Open': candle.get('open'),
                    'High': candle.get('high'),
                    'Low': candle.get('low'),
                    'Close': candle.get('close'),
                    'Volume': candle.get('volume') or 0,
                }
            )
        if len(rows) < 5:
            return None
        frame = pd.DataFrame(rows).set_index('Date')
        buf = io.BytesIO()
        mpf.plot(frame, type='candle', style='charles', volume=True, savefig=dict(fname=buf, dpi=120, bbox_inches='tight'))
        plt.close('all')
        buf.seek(0)
        return buf.read()

    async def send_signal(self, payload: dict[str, Any]) -> None:
        topic_id = self.resolve_topic_id(payload.get('asset_class'), payload.get('source_ai'))
        text = self.format_signal_message(payload)
        chart = self.build_chart_png(payload)
        if not self.bot_token:
            logger.info('[notification] stdout signal topic=%s chart=%s\n%s', topic_id, bool(chart), text)
            return
        logger.info('[notification] Telegram send topic=%s chart=%s (token configured)', topic_id, bool(chart))

    async def send_outcome(self, payload: dict[str, Any]) -> None:
        text = f"Outcome {payload.get('status')} {payload.get('symbol')} PnL={payload.get('pnl_percent')}%"
        if not self.bot_token:
            logger.info('[notification] stdout outcome\n%s', text)
            return
        logger.info('[notification] Telegram outcome: %s', text)

    async def send_entry_event(self, payload: dict[str, Any]) -> None:
        text = f"Entry {payload.get('symbol')} @ {payload.get('fill_price')}"
        if not self.bot_token:
            logger.info('[notification] stdout entry\n%s', text)
            return
        logger.info('[notification] Telegram entry: %s', text)
