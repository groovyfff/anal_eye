from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class OrmBase(DeclarativeBase):
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class SignalFeatureLog(OrmBase):
    """Центральная таблица сигналов — crypto и external markets."""

    __tablename__ = 'signal_feature_logs'
    __table_args__ = (Index('ix_signal_feature_logs_symbol_ts', 'symbol', 'initial_metric_timestamp'),)

    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, server_default='crypto')
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signal_id_uuid: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    initial_metric_timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trigger_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heuristic_signal_consensus: Mapped[str | None] = mapped_column(String(16), nullable=True)
    composite_score_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    feat_current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_price_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_market_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feat_rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_macd_hist: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_ema_short: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_ema_long: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_ema_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_ema_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_bb_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_bb_middle: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_bb_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_bb_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_atr_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_vol_rel: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_support_nearest: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_resistance_nearest: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_sp500_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_dxy_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_bid_ask_spread_pips: Mapped[float | None] = mapped_column(Float, nullable=True)

    feat_funding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_open_interest_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_liquidations_long_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_liquidations_short_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    feat_cvd: Mapped[float | None] = mapped_column(Float, nullable=True)

    ohlcv_open_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ohlcv_high_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ohlcv_low_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ohlcv_close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ohlcv_volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    strat_indicators_consensus: Mapped[str | None] = mapped_column(String(16), nullable=True)
    strat_patterns_consensus: Mapped[str | None] = mapped_column(String(16), nullable=True)
    historical_snapshots_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    historical_ohlcv_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    indicators_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    patterns_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    features_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    ai_signal_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_reason_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_entry_price_suggestion: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_tp_price_suggestion: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_sl_price_suggestion: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_leverage_suggestion: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_consensus_achieved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    telegram_message_sent: Mapped[bool | None] = mapped_column(Boolean, nullable=True, server_default='false')

    tracker_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tracker_entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tracker_exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tracker_pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    tracker_pnl_usdt: Mapped[float | None] = mapped_column(Float, nullable=True)
    tracker_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tracker_closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EnsembleModelDecision(OrmBase):
    __tablename__ = 'ensemble_model_decisions'

    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, server_default='crypto')
    signal_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)


class EnsembleBacktestResult(OrmBase):
    __tablename__ = 'ensemble_backtest_results'

    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, server_default='crypto')
    signal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pnl_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Trade(OrmBase):
    __tablename__ = 'trades'

    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, server_default='crypto')
    signal_id_uuid: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    leverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default='active')
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_telegram_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    opened_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
