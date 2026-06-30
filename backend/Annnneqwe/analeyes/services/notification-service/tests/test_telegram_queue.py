from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

# Keep tests lightweight — chart deps are not needed for queue behavior.
sys.modules.setdefault("matplotlib", MagicMock())
sys.modules.setdefault("matplotlib.pyplot", MagicMock())
sys.modules.setdefault("mplfinance", MagicMock())

import src.logic.telegram.telegram_sender as telegram_mod
import pytest

from src.logic.telegram.telegram_sender import TelegramSender


def test_telegram_send_worker_serializes_and_waits() -> None:
    sleeps: list[float] = []
    calls: list[str] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def _fake_post(method: str, payload: dict) -> dict:
        calls.append(method)
        return {"ok": True, "result": {"message_id": len(calls)}}

    original_sleep = telegram_mod.asyncio.sleep
    telegram_mod.asyncio.sleep = _fake_sleep  # type: ignore[assignment]

    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    sender.bot_token = "test-token"
    sender.group_id = -100123
    sender.allow_no_topic = True
    sender._post_telegram_direct = _fake_post  # type: ignore[method-assign]
    sender._send_interval_ms = 200

    async def _run() -> None:
        sender.start_send_worker()
        assert sender._send_worker_task is not None
        try:
            await asyncio.gather(
                sender._enqueue_post_telegram("sendMessage", {"chat_id": 1, "text": "a"}),
                sender._enqueue_post_telegram("sendMessage", {"chat_id": 1, "text": "b"}),
                sender._enqueue_post_telegram("sendMessage", {"chat_id": 1, "text": "c"}),
            )
        finally:
            sender._send_worker_task.cancel()
            try:
                await sender._send_worker_task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(_run())
    finally:
        telegram_mod.asyncio.sleep = original_sleep

    assert calls == ["sendMessage", "sendMessage", "sendMessage"]
    assert sleeps == [0.2, 0.2, 0.2]


def test_telegram_signal_message_uses_dynamic_symbol() -> None:
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    text = sender._format_signal_message(
        {
            "symbol": "ETHUSDT",
            "asset": "ETH",
            "asset_class": "crypto",
            "decision": "LONG",
            "confidence": 0.8,
            "entry_price": 3000,
            "tp": 3100,
            "sl": 2900,
        }
    )
    assert "ETHUSDT" in text
    assert "ETH" in text
    assert "🟢" in text


def test_telegram_skip_message_format() -> None:
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    text = sender._format_signal_message(
        {
            "symbol": "ETHUSDT",
            "asset": "ETH",
            "asset_class": "crypto",
            "decision": "SKIP",
            "confidence": 0.41,
            "skip_reason": "confidence_below_threshold",
        }
    )
    assert "⚪" in text
    assert "SKIP" in text
    assert "confidence_below_threshold" in text


def test_telegram_group_id_preserves_negative_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_GROUP_ID", "-1004293390337")
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    assert sender.group_id == -1004293390337


def test_should_notify_skipped_decisions_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTIFICATION_SEND_SKIPPED_DECISIONS", "false")
    sender = TelegramSender({"telegram": {"retry_attempts": 1}})
    assert sender.should_notify({"symbol": "ETHUSDT", "decision": "SKIP"}) is False

    monkeypatch.setenv("NOTIFICATION_SEND_SKIPPED_DECISIONS", "true")
    sender_on = TelegramSender({"telegram": {"retry_attempts": 1}})
    assert sender_on.should_notify({"symbol": "ETHUSDT", "decision": "SKIP"}) is True
