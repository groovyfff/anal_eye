from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import SignalFeatureLog

logger = logging.getLogger(__name__)

FEATURE_COLUMN_MAP = {
    'current_price': 'feat_current_price',
    'price_pct': 'feat_price_pct',
    'market_state': 'feat_market_state',
    'rsi': 'feat_rsi',
    'macd': 'feat_macd',
    'macd_signal': 'feat_macd_signal',
    'macd_hist': 'feat_macd_hist',
    'ema_short': 'feat_ema_short',
    'ema_long': 'feat_ema_long',
    'ema_50': 'feat_ema_50',
    'ema_200': 'feat_ema_200',
    'adx': 'feat_adx',
    'bb_upper': 'feat_bb_upper',
    'bb_middle': 'feat_bb_middle',
    'bb_lower': 'feat_bb_lower',
    'bb_width': 'feat_bb_width',
    'atr': 'feat_atr',
    'atr_pct': 'feat_atr_pct',
    'vwap': 'feat_vwap',
    'vol_rel': 'feat_vol_rel',
    'obv': 'feat_obv',
    'support_nearest': 'feat_support_nearest',
    'resistance_nearest': 'feat_resistance_nearest',
    'sp500_correlation': 'feat_sp500_correlation',
    'dxy_correlation': 'feat_dxy_correlation',
    'bid_ask_spread_pips': 'feat_bid_ask_spread_pips',
}


async def save_external_candidate_log(
    session: AsyncSession,
    payload: dict[str, Any],
) -> int:
    """Сохраняет строку signal_feature_logs для external-markets кандидата."""
    features = payload.get('features') or {}
    signal_id = uuid.UUID(str(payload['signal_id']))
    timestamp_raw = payload['timestamp']
    if isinstance(timestamp_raw, str):
        initial_ts = dt.datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
    else:
        initial_ts = timestamp_raw
    if initial_ts.tzinfo is None:
        initial_ts = initial_ts.replace(tzinfo=dt.timezone.utc)

    row = SignalFeatureLog(
        asset_class=str(payload['asset_class']),
        symbol=str(payload['symbol']),
        display_name=payload.get('name'),
        signal_id_uuid=signal_id,
        initial_metric_timestamp=initial_ts,
        trigger_reason=payload.get('trigger_reason'),
        heuristic_signal_consensus=payload.get('heuristic_signal_consensus'),
        composite_score_value=payload.get('composite_score'),
        strat_indicators_consensus=(payload.get('indicators') or {}).get('consensus'),
        strat_patterns_consensus=(payload.get('patterns') or {}).get('consensus'),
        historical_snapshots_json=payload.get('historical_snapshots'),
        historical_ohlcv_json=payload.get('historical_ohlcv'),
        indicators_json=payload.get('indicators'),
        patterns_json=payload.get('patterns'),
        features_json=features,
        ohlcv_close_price=features.get('current_price'),
    )
    for feat_key, col_name in FEATURE_COLUMN_MAP.items():
        setattr(row, col_name, features.get(feat_key))

    ohlcv = payload.get('historical_ohlcv') or []
    if ohlcv:
        last_candle = ohlcv[-1]
        row.ohlcv_open_price = last_candle.get('open')
        row.ohlcv_high_price = last_candle.get('high')
        row.ohlcv_low_price = last_candle.get('low')
        row.ohlcv_close_price = last_candle.get('close')
        row.ohlcv_volume = last_candle.get('volume')

    session.add(row)
    await session.flush()
    db_id = int(row.id)
    logger.info('[external-markets] signal_feature_logs id=%s symbol=%s asset_class=%s', db_id, row.symbol, row.asset_class)
    return db_id
