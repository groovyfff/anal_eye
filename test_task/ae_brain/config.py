"""Centralised, environment-driven configuration for A.E. Brain.

All tunables live here so that the live trading loop, the trainers and the CLI
share a single, validated source of truth. Values can be overridden via
environment variables (prefix ``AEB_``) or a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_allowed_symbols_csv() -> str:
    from ae_brain.symbols import default_allowed_symbols_csv

    return default_allowed_symbols_csv()


class DatabaseConfig(BaseSettings):
    """Async PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="AEB_DB_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "ae_brain"
    password: str = "ae_brain"
    name: str = "ae_brain"
    min_pool_size: int = 2
    max_pool_size: int = 16
    command_timeout: float = 30.0

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        return self.dsn


class AmqpInputConfig(BaseSettings):
    """RabbitMQ consumer settings for ``data.candidates.ai``."""

    model_config = SettingsConfigDict(env_prefix="AEB_INPUT_", extra="ignore")

    amqp_url: str = ""
    exchange: str = "analeyes.events"
    queue: str = "q_data_candidates_ai"
    routing_key: str = "data.candidates.ai"
    prefetch_count: int = 16
    consumer_tag: str = "ae-brain-q_data_candidates_ai"
    requeue_on_error: bool = True

    @property
    def resolved_url(self) -> str:
        return resolve_amqp_url(self.amqp_url)


class AmqpOutputConfig(BaseSettings):
    """RabbitMQ publisher settings for ``signal.final``."""

    model_config = SettingsConfigDict(env_prefix="AEB_OUTPUT_", extra="ignore")

    amqp_url: str = ""
    exchange: str = "analeyes.events"
    routing_key: str = "signal.final"

    @property
    def resolved_url(self) -> str:
        return resolve_amqp_url(self.amqp_url)


class LegacyAmqpConfig(BaseSettings):
    """Deprecated single-broker settings kept for backward compatibility."""

    model_config = SettingsConfigDict(env_prefix="AEB_AMQP_", extra="ignore")

    url: str = ""
    host: str = "host.docker.internal"
    port: int = 5672


class TelegramDebugConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, validation_alias="AEB_DIRECT_TELEGRAM_ENABLED")
    bot_token: str = Field(default="", validation_alias="TELEGRAM_BOT_TOKEN")
    group_id: str = Field(default="", validation_alias="TELEGRAM_GROUP_ID")
    topic_id: str | None = Field(default=None, validation_alias="TELEGRAM_TOPIC_ID")


def resolve_amqp_url(
    explicit_url: str,
    *,
    legacy_url: str = "",
    host: str = "host.docker.internal",
) -> str:
    """Resolve AMQP URL with analeyes vhost defaults (never guest/%2F)."""
    import os

    if explicit_url:
        return explicit_url
    if legacy_url:
        return legacy_url
    password = (
        os.getenv("RABBITMQ_APP_PASSWORD")
        or os.getenv("RABBITMQ_PASSWORD")
        or os.getenv("RABBITMQ_PASS")
        or os.getenv("AEB_RABBITMQ_APP_PASSWORD")
        or "analeyes_dev_secret"
    )
    return f"amqp://analeyes:{password}@{host}:5672/analeyes"


# Backward-compatible alias used by older imports/tests.
class RabbitMQConfig(AmqpInputConfig):
    """Deprecated alias; prefer :class:`AmqpInputConfig`."""

    consume_queue: str = "q_data_candidates_ai"
    publish_exchange: str = "analeyes.events"
    publish_routing_key: str = "signal.final"

    def __init__(self, **data: object) -> None:
        if "consume_queue" in data and "queue" not in data:
            data["queue"] = data.pop("consume_queue")
        if "publish_exchange" in data and "exchange" not in data:
            data["exchange"] = data.pop("publish_exchange")
        if "publish_routing_key" in data and "routing_key" not in data:
            data["routing_key"] = data.pop("publish_routing_key")
        if "url" in data and "amqp_url" not in data:
            data["amqp_url"] = data.pop("url")
        super().__init__(**data)


class GPUConfig(BaseSettings):
    """Hardware / precision settings for the 4x Tesla P100 (Pascal) target."""

    model_config = SettingsConfigDict(env_prefix="AEB_GPU_", extra="ignore")

    enabled: bool = True
    # Round-robin device assignment across the 4 P100s.
    device_ids: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    # P100 has hardware fp16; we run inference in half precision.
    use_fp16: bool = True
    # Prefer ONNXRuntime-GPU for sequence inference when an exported model exists.
    prefer_onnx: bool = True


class ExecutorConfig(BaseSettings):
    """Thread / process pool sizing for offloaded inference."""

    model_config = SettingsConfigDict(env_prefix="AEB_EXEC_", extra="ignore")

    # Torch/ONNX releases the GIL during inference -> threads are sufficient and
    # avoid CUDA-context-per-process overhead.
    thread_workers: int = 8
    # CPU-bound feature engineering / gradient boosting can use processes.
    process_workers: int = 4


class CostConfig(BaseSettings):
    """Binance-derived transaction cost model parameters (USD-M futures)."""

    model_config = SettingsConfigDict(env_prefix="AEB_COST_", extra="ignore")

    taker_fee_rate: float = 0.0004  # 4 bps taker
    maker_fee_rate: float = 0.0002  # 2 bps maker
    # Funding is charged every 8h; we annualise per-trade by expected holding.
    default_funding_rate_8h: float = 0.0001
    # Slippage modelled as a function of notional / book depth; base in bps.
    base_slippage_bps: float = 1.5
    slippage_impact_coeff: float = 0.35  # extra bps per 1x ADV participation


class RiskConfig(BaseSettings):
    """Position sizing and portfolio risk constraints."""

    model_config = SettingsConfigDict(env_prefix="AEB_RISK_", extra="ignore")

    account_equity_usd: float = 100_000.0
    max_leverage: float = 5.0
    # Fractional Kelly: never bet full Kelly (variance + estimation error).
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.20  # cap any single position at 20% equity
    min_position_pct: float = 0.005
    # ATR-based stop distance multipliers (dynamic, NOT hardcoded -5%).
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    # Correlation limit: reject if summed |corr| exposure exceeds this budget.
    max_correlated_exposure: float = 1.5
    correlation_threshold: float = 0.6


class ModelConfig(BaseSettings):
    """Model architecture / artifact locations."""

    model_config = SettingsConfigDict(env_prefix="AEB_MODEL_", extra="ignore")

    artifacts_dir: Path = Path("artifacts")
    tabular_backend: Literal["lightgbm", "xgboost", "catboost"] = "lightgbm"
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"  # sigmoid=Platt
    sequence_backend: Literal["lstm", "gru", "patchtst"] = "patchtst"
    sequence_window: int = 48  # >= 30 candles required
    rl_algo: Literal["ppo", "sac"] = "ppo"
    # Unsupervised market-regime detector (GaussianMixture) + meta stacker.
    n_regimes: int = 3
    regime_enabled: bool = True
    use_meta_model: bool = True
    meta_prefer_two_stage: bool = True

    @field_validator("sequence_window")
    @classmethod
    def _min_window(cls, v: int) -> int:
        if v < 30:
            raise ValueError("sequence_window must be >= 30 candles (spec requirement)")
        return v


class FusionConfig(BaseSettings):
    """Fusion-layer aggregation weights and decision thresholds."""

    model_config = SettingsConfigDict(env_prefix="AEB_FUSION_", extra="ignore")

    w_tabular: float = 0.45
    w_sequence: float = 0.30
    w_rl: float = 0.25
    # Minimum fused directional conviction to consider a trade at all.
    min_conviction: float = 0.55
    # Minimum positive EV (USD) below which we SKIP even if EV>0 (noise floor).
    min_ev_usd: float = 0.0
    # Meta-model thresholds (two-stage + legacy).
    meta_direction_threshold: float = 0.30
    meta_trade_threshold: float = 0.45
    meta_direction_margin: float = 0.05
    meta_mode: Literal["two_stage", "legacy_3class", "side_aware_ensemble", "side_specialists"] = "two_stage"
    # Downweight / mask weak base layers in meta features.
    min_sequence_val_auc: float = 0.52
    min_rl_mean_reward: float = 0.0


class Settings(BaseSettings):
    """Top-level settings aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="AEB_", env_file=".env", extra="ignore"
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    allow_legacy_guest_vhost: bool = Field(default=False, validation_alias="AEB_ALLOW_LEGACY_GUEST_VHOST")
    min_composite_score: float = Field(default=0.0, validation_alias="AEB_MIN_COMPOSITE_SCORE")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    amqp_input: AmqpInputConfig = Field(default_factory=AmqpInputConfig)
    amqp_output: AmqpOutputConfig = Field(default_factory=AmqpOutputConfig)
    amqp_legacy: LegacyAmqpConfig = Field(default_factory=LegacyAmqpConfig)
    telegram_debug: TelegramDebugConfig = Field(default_factory=TelegramDebugConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    enable_chromadb_rag: bool = False
    # News sentiment features consumed from RabbitMQ (data.news.sentiment).
    # Echo-only: attaches the latest fresh snapshot to candidate.meta; does not
    # yet change trade decisions (see messaging/news_features.py + DECISIONS.md).
    enable_news_features: bool = False
    news_features_amqp_url: str = Field(default="", validation_alias="AEB_NEWS_AMQP_URL")
    news_features_queue: str = Field(
        default="q_data_news_sentiment", validation_alias="AEB_NEWS_FEATURES_QUEUE"
    )
    news_features_routing_key: str = Field(
        default="data.news.sentiment", validation_alias="AEB_NEWS_FEATURES_ROUTING_KEY"
    )
    news_features_exchange: str = Field(
        default="analeyes.events", validation_alias="AEB_NEWS_FEATURES_EXCHANGE"
    )
    news_features_max_age_s: float = Field(
        default=300.0, validation_alias="AEB_NEWS_FEATURES_MAX_AGE_S"
    )
    # --- Optional news.market_signal fusion (OpenRouter path) ----------------
    # Consumes per-item news market signals (score 1-10) and applies a BOUNDED
    # confidence/EV nudge. Disabled-by-default-capable but on by default; it is
    # a pure no-op when the queue is empty or the broker is unreachable.
    news_signal_enabled: bool = Field(default=True, validation_alias="AEB_NEWS_SIGNAL_ENABLED")
    news_signal_exchange: str = Field(
        default="analeyes.events", validation_alias="AEB_NEWS_SIGNAL_EXCHANGE"
    )
    news_signal_queue: str = Field(
        default="q_news_market_signal", validation_alias="AEB_NEWS_SIGNAL_QUEUE"
    )
    news_signal_routing_key: str = Field(
        default="news.market_signal", validation_alias="AEB_NEWS_SIGNAL_ROUTING_KEY"
    )
    news_signal_ttl_sec: float = Field(
        default=21600.0, validation_alias="AEB_NEWS_SIGNAL_TTL_SEC"
    )
    news_max_conf_delta: float = Field(
        default=0.05, validation_alias="AEB_NEWS_MAX_CONF_DELTA"
    )
    news_max_ev_multiplier_delta: float = Field(
        default=0.10, validation_alias="AEB_NEWS_MAX_EV_MULTIPLIER_DELTA"
    )
    news_min_relevance: float = Field(
        default=0.65, validation_alias="AEB_NEWS_MIN_RELEVANCE"
    )
    news_signal_amqp_url: str = Field(default="", validation_alias="AEB_NEWS_SIGNAL_AMQP_URL")
    publish_skipped_decisions: bool = Field(
        default=False, validation_alias="AEB_PUBLISH_SKIPPED_DECISIONS"
    )
    disable_signal_dedup_in_test_mode: bool = Field(
        default=False, validation_alias="AEB_DISABLE_SIGNAL_DEDUP_IN_TEST_MODE"
    )
    allowed_symbols: str = Field(
        default_factory=_default_allowed_symbols_csv,
        validation_alias="AEB_ALLOWED_SYMBOLS",
    )
    min_publish_confidence: float = Field(default=0.70, validation_alias="AEB_MIN_PUBLISH_CONFIDENCE")
    only_btc: bool = Field(default=False, validation_alias="AEB_ONLY_BTC")

    @property
    def allowed_symbol_set(self) -> frozenset[str]:
        from ae_brain.messaging.publish_gate import parse_allowed_symbols

        return parse_allowed_symbols(self.allowed_symbols, only_btc=self.only_btc)

    @model_validator(mode="after")
    def _resolve_amqp_urls(self) -> "Settings":
        host = self.amqp_legacy.host
        legacy_url = self.amqp_legacy.url
        input_url = self.amqp_input.amqp_url or resolve_amqp_url("", legacy_url=legacy_url, host=host)
        output_url = self.amqp_output.amqp_url or input_url
        if input_url != self.amqp_input.amqp_url:
            self.amqp_input = self.amqp_input.model_copy(update={"amqp_url": input_url})
        if output_url != self.amqp_output.amqp_url:
            self.amqp_output = self.amqp_output.model_copy(update={"amqp_url": output_url})
        return self

    @model_validator(mode="after")
    def _forbid_unsafe_production_flags(self) -> "Settings":
        if self.env in ("prod", "staging"):
            if self.publish_skipped_decisions:
                raise ValueError(
                    "AEB_PUBLISH_SKIPPED_DECISIONS=true is forbidden when AEB_ENV is prod/staging"
                )
            if self.telegram_debug.enabled:
                raise ValueError(
                    "AEB_DIRECT_TELEGRAM_ENABLED=true is forbidden when AEB_ENV is prod/staging "
                    "(Telegram must consume only signal.final)"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton settings instance."""
    return Settings()
