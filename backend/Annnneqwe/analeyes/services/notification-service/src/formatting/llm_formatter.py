from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from src.formatting.deterministic_formatter import (
    DeterministicSignalFormatter,
    SignalFacts,
    validate_llm_comment,
)
from src.formatting.presentation_config import PresentationConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Telegram signal presentation formatter. You cannot change trading decisions, "
    "numbers, direction, prices, confidence, risk, or model output. Return ONLY JSON object: "
    '{"comment":"..."}. No markdown. No prose. Russian language only. The comment must be short, '
    "natural, and based only on provided facts."
)


def strip_api_key_for_log(api_key: str) -> str:
    key = api_key.strip().strip('"').strip("'")
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : idx + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


class OpenRouterCommentFormatter:
    """Optional LLM comment generator — presentation only, never mutates signal facts."""

    def __init__(
        self,
        config: PresentationConfig | None = None,
        *,
        deterministic: DeterministicSignalFormatter | None = None,
    ) -> None:
        self._config = config or PresentationConfig.from_env()
        self._deterministic = deterministic or DeterministicSignalFormatter()

    @property
    def config(self) -> PresentationConfig:
        return self._config

    async def generate_comment(self, facts: SignalFacts) -> str:
        if not self._config.llm_enabled:
            return self._deterministic.build_fallback_comment(facts)
        if not self._config.api_key:
            logger.info("signal_formatter_llm_disabled_missing_key symbol=%s", facts.symbol)
            return self._deterministic.build_fallback_comment(facts)

        sanitized = facts.to_sanitized_dict()
        last_error: str | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                logger.info(
                    "signal_formatter_llm_request_started symbol=%s model=%s attempt=%s",
                    facts.symbol,
                    self._config.model,
                    attempt,
                )
                raw_comment = await self._request_comment(sanitized)
                parsed = extract_json_object(raw_comment)
                if not parsed or not isinstance(parsed.get("comment"), str):
                    last_error = "invalid_json"
                    continue
                comment = str(parsed["comment"]).strip()
                rejection = validate_llm_comment(
                    comment,
                    facts,
                    max_chars=self._config.max_comment_chars,
                )
                if rejection:
                    logger.info(
                        "signal_formatter_comment_rejected symbol=%s reason=%s",
                        facts.symbol,
                        rejection,
                    )
                    last_error = rejection
                    continue
                logger.info(
                    "signal_formatter_llm_comment_generated symbol=%s chars=%s",
                    facts.symbol,
                    len(comment),
                )
                return comment
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "signal_formatter_llm_failed symbol=%s attempt=%s err=%s",
                    facts.symbol,
                    attempt,
                    exc,
                )
        logger.info(
            "signal_formatter_llm_failed symbol=%s reason=%s",
            facts.symbol,
            last_error or "unknown",
        )
        return self._deterministic.build_fallback_comment(facts)

    async def _request_comment(self, sanitized_facts: dict[str, Any]) -> str:
        url = f"{self._config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://analeyes.local",
            "X-Title": "AnalEyes Signal Formatter",
        }
        payload = {
            "model": self._config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(sanitized_facts, ensure_ascii=False)},
            ],
        }
        async with httpx.AsyncClient(timeout=self._config.timeout_sec) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("empty_choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise ValueError("empty_content")
        return str(content)

    def build_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://analeyes.local",
            "X-Title": "AnalEyes Signal Formatter",
        }
