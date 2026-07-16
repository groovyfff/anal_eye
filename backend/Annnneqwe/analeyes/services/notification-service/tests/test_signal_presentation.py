from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.formatting.deterministic_formatter import (
    DeterministicSignalFormatter,
    TELEGRAM_CAPTION_LIMIT,
    extract_signal_facts,
    validate_llm_comment,
)
from src.formatting.llm_formatter import OpenRouterCommentFormatter, extract_json_object, strip_api_key_for_log
from src.formatting.presentation_config import PresentationConfig
from src.logic.telegram.telegram_sender import TelegramSender


def _long_payload(**overrides) -> dict:
    base = {
        "symbol": "BTCUSDT",
        "decision": "LONG",
        "confidence": 0.70,
        "market_state": "TREND",
        "model": "meta-llama/llama-4-maverick",
        "entry_price": 67123.45,
        "sl": 66200.0,
        "tp": 69000.0,
        "leverage": 3,
        "features": {"trend_impulse_10m": -9},
        "signal_time": "2026-02-02T05:29:40Z",
    }
    base.update(overrides)
    return base


def _short_payload(**overrides) -> dict:
    return _long_payload(decision="SHORT", **overrides)


def _sample_candles(n: int = 80) -> list[dict]:
    candles = []
    price = 67000.0
    for i in range(n):
        open_p = price
        close_p = price + (1 if i % 2 == 0 else -1) * 10
        candles.append(
            {
                "timestamp": f"2026-02-01T{i % 24:02d}:00:00Z",
                "open": open_p,
                "high": max(open_p, close_p) + 5,
                "low": min(open_p, close_p) - 5,
                "close": close_p,
                "volume": 100 + i,
            }
        )
        price = close_p
    return candles


@pytest.fixture
def formatter() -> DeterministicSignalFormatter:
    return DeterministicSignalFormatter()


def test_llm_formatter_cannot_change_direction(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_long_payload())
    rejection = validate_llm_comment("Сильный шорт по BTCUSDT", facts, max_chars=220)
    assert rejection == "contradicts_direction"


def test_llm_formatter_cannot_change_confidence(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_long_payload(confidence=0.70))
    rejection = validate_llm_comment("Уверенность 95% выглядит убедительно", facts, max_chars=220)
    assert rejection == "fabricated_confidence"


def test_llm_formatter_cannot_change_symbol(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_long_payload())
    rejection = validate_llm_comment("ETHUSDT выглядит сильнее", facts, max_chars=220)
    assert rejection == "foreign_symbol"


def test_bad_invalid_llm_json_falls_back() -> None:
    assert extract_json_object("not json at all") is None
    assert extract_json_object('{"wrong":"field"}') == {"wrong": "field"}


@pytest.mark.asyncio
async def test_missing_api_key_falls_back_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNAL_PRESENTATION_LLM_ENABLED", "true")
    monkeypatch.delenv("SIGNAL_PRESENTATION_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = PresentationConfig.from_env()
    llm = OpenRouterCommentFormatter(config)
    facts = extract_signal_facts(_long_payload())
    comment = await llm.generate_comment(facts)
    assert comment
    assert "покупател" in comment.lower() or "риск" in comment.lower()


def test_openrouter_headers_include_required_fields() -> None:
    config = PresentationConfig(
        llm_enabled=True,
        provider="openrouter",
        model="x-ai/grok-4.3",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test-key",
        timeout_sec=15,
        max_retries=2,
        max_comment_chars=220,
    )
    llm = OpenRouterCommentFormatter(config)
    headers = llm.build_request_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["HTTP-Referer"] == "https://analeyes.local"
    assert headers["X-Title"] == "AnalEyes Signal Formatter"


def test_api_key_is_stripped_and_never_logged() -> None:
    assert strip_api_key_for_log('  "sk-or-v1-abcdefgh12345678"  ') == "sk-o...5678"
    assert strip_api_key_for_log("") == ""
    assert "sk-or-v1-abcdefgh12345678" not in strip_api_key_for_log("sk-or-v1-abcdefgh12345678")


def test_long_caption_matches_target_format(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_long_payload())
    comment = "Короткий комментарий."
    caption = formatter.build_caption(facts, comment)
    assert caption.startswith("🚀 AnalEyes Рекомендация")
    assert "💎 Монета: BTCUSDT" in caption
    assert "🧠 Модель AI: meta-llama/llama-4-maverick" in caption
    assert "🎯 Направление: 🟢 Лонг" in caption
    assert "🔥 Уверенность AI: 70%" in caption
    assert "📊 Состояние Рынка: TREND" in caption
    assert "📈 Тренд-импульс (10m): -9" in caption
    assert "📉 Вход (предполагаемый): 67123.45 USDT" in caption
    assert "💰 Стоп-Лосс (SL): 66200.00 USDT" in caption
    assert "🏆 Тейк-Профит (TP): 69000.00 USDT" in caption
    assert "🧮 Плечо (макс): x3" in caption
    assert "🔗 Торговать BTCUSDT на Binance Futures" in caption
    assert "https://www.binance.com/en/futures/BTCUSDT" not in caption
    assert "🕘 02.02.2026 05:29:40 UTC" in caption


def test_short_caption_matches_target_format(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_short_payload())
    caption = formatter.build_caption(facts, "Комментарий.")
    assert "🎯 Направление: 🔴 Шорт" in caption


def test_missing_entry_sl_tp_leverage_renders_na(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(
        _long_payload(entry_price=None, sl=None, tp=None, leverage=None)
    )
    caption = formatter.build_caption(facts, "Комментарий.")
    assert "📉 Вход (предполагаемый): N/A USDT" in caption
    assert "💰 Стоп-Лосс (SL): N/A USDT" in caption
    assert "🏆 Тейк-Профит (TP): N/A USDT" in caption
    assert "🧮 Плечо (макс): xN/A" in caption


def test_chart_renderer_creates_png_when_candles_exist() -> None:
    pytest.importorskip("mplfinance")
    import matplotlib

    matplotlib.use("Agg")
    from src.formatting.chart_renderer import ChartRenderer

    renderer = ChartRenderer()
    payload = _long_payload(candles=_sample_candles())
    image = renderer.render(payload)
    assert image is not None
    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_renderer_skips_gracefully_when_candles_missing() -> None:
    from src.formatting.chart_renderer import ChartRenderer

    renderer = ChartRenderer()
    assert renderer.render(_long_payload()) is None


@pytest.mark.asyncio
async def test_telegram_sender_uses_send_photo_when_chart_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOW_NO_TOPIC", "true")
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    sender.bot_token = "token"
    sender.group_id = -100
    sender.allow_no_topic = True

    calls: list[tuple[str, dict, dict | None]] = []

    async def _fake_enqueue(method: str, payload: dict, *, files: dict | None = None) -> dict:
        calls.append((method, payload, files))
        return {"ok": True}

    sender._enqueue_post_telegram = _fake_enqueue  # type: ignore[method-assign]
    sender._presenter.present = AsyncMock(  # type: ignore[method-assign]
        return_value=type("P", (), {"caption": "caption", "chart_bytes": b"png"})()
    )

    result = await sender.send_signal(_long_payload(confidence=0.75))
    assert result.sent is True
    assert calls[0][0] == "sendPhoto"
    assert calls[0][2] is not None


@pytest.mark.asyncio
async def test_telegram_sender_uses_send_message_when_chart_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOW_NO_TOPIC", "true")
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    sender.bot_token = "token"
    sender.group_id = -100
    sender.allow_no_topic = True

    calls: list[str] = []

    async def _fake_enqueue(method: str, payload: dict, *, files: dict | None = None) -> dict:
        calls.append(method)
        return {"ok": True}

    sender._enqueue_post_telegram = _fake_enqueue  # type: ignore[method-assign]
    sender._presenter.present = AsyncMock(  # type: ignore[method-assign]
        return_value=type("P", (), {"caption": "caption", "chart_bytes": None})()
    )

    result = await sender.send_signal(_long_payload(confidence=0.75))
    assert result.sent is True
    assert calls == ["sendMessage"]


@pytest.mark.asyncio
async def test_skip_signal_is_not_sent() -> None:
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    sender.bot_token = "token"
    sender.group_id = -100
    sender.allow_no_topic = True
    sender._enqueue_post_telegram = AsyncMock()  # type: ignore[method-assign]

    result = await sender.send_signal(
        {"symbol": "BTCUSDT", "decision": "SKIP", "confidence": 0.95, "source_ai": "ae_brain"}
    )
    assert result.sent is False
    sender._enqueue_post_telegram.assert_not_called()


def test_llm_comment_with_contradictory_direction_is_rejected() -> None:
    facts = extract_signal_facts(_short_payload())
    assert validate_llm_comment("Отличный лонг для входа", facts, max_chars=220) == "contradicts_direction"


def test_llm_comment_with_fabricated_price_is_rejected() -> None:
    facts = extract_signal_facts(_long_payload())
    assert validate_llm_comment("Цель около 99999.99 выглядит реалистично", facts, max_chars=220) == "fabricated_price"


def test_full_caption_length_is_telegram_safe(formatter: DeterministicSignalFormatter) -> None:
    facts = extract_signal_facts(_long_payload())
    long_comment = "А" * 500
    caption = formatter.build_caption(facts, long_comment)
    truncated = formatter.truncate_caption_for_telegram(caption)
    assert len(truncated) <= TELEGRAM_CAPTION_LIMIT


@pytest.mark.asyncio
async def test_openrouter_valid_json_comment_used(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PresentationConfig(
        llm_enabled=True,
        provider="openrouter",
        model="x-ai/grok-4.3",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        timeout_sec=5,
        max_retries=1,
        max_comment_chars=220,
    )
    llm = OpenRouterCommentFormatter(config)
    facts = extract_signal_facts(_long_payload())

    async def _fake_request(_: dict) -> str:
        return json.dumps({"comment": "Рынок сохраняет бычий импульс при умеренной волатильности."})

    llm._request_comment = _fake_request  # type: ignore[method-assign]
    comment = await llm.generate_comment(facts)
    assert "бычий" in comment.lower()


def test_extract_json_object_strips_fences() -> None:
    raw = '```json\n{"comment":"тест"}\n```'
    parsed = extract_json_object(raw)
    assert parsed == {"comment": "тест"}
