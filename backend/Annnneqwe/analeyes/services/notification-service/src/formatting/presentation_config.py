from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _strip_api_key(value: str) -> str:
    return value.strip().strip('"').strip("'")


@dataclass(frozen=True, slots=True)
class PresentationConfig:
    llm_enabled: bool
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_sec: float
    max_retries: int
    max_comment_chars: int

    @classmethod
    def from_env(cls) -> "PresentationConfig":
        api_key = _strip_api_key(
            os.environ.get("SIGNAL_PRESENTATION_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
        base_url = (
            os.environ.get("SIGNAL_PRESENTATION_BASE_URL", "").strip()
            or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        ).rstrip("/")
        model = (
            os.environ.get("SIGNAL_PRESENTATION_MODEL", "").strip()
            or os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.3").strip()
        )
        return cls(
            llm_enabled=_env_bool("SIGNAL_PRESENTATION_LLM_ENABLED", default=False),
            provider=os.environ.get("SIGNAL_PRESENTATION_PROVIDER", "openrouter").strip() or "openrouter",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_sec=_env_float(
                "SIGNAL_PRESENTATION_TIMEOUT_SEC",
                _env_float("OPENROUTER_TIMEOUT_SEC", 15.0),
            ),
            max_retries=_env_int(
                "SIGNAL_PRESENTATION_MAX_RETRIES",
                _env_int("OPENROUTER_MAX_RETRIES", 2),
            ),
            max_comment_chars=_env_int("SIGNAL_PRESENTATION_MAX_COMMENT_CHARS", 220),
        )

    @property
    def llm_ready(self) -> bool:
        return self.llm_enabled and bool(self.api_key)
