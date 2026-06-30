"""Optional debug-only direct Telegram notifications (default disabled)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ae_brain.config import TelegramDebugConfig
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.telegram_debug")


async def send_debug_telegram(cfg: TelegramDebugConfig, payload: dict[str, Any]) -> None:
    if not cfg.enabled:
        return
    if not cfg.bot_token or not cfg.group_id:
        log.warning("telegram_debug.missing_config")
        return
    symbol = payload.get("symbol", "?")
    decision = payload.get("decision", "?")
    text = f"[AE Brain debug] {symbol} {decision}\n{json.dumps(payload, indent=2)[:3500]}"
    url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
    body: dict[str, Any] = {"chat_id": cfg.group_id, "text": text}
    if cfg.topic_id:
        body["message_thread_id"] = int(cfg.topic_id)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        log.info("telegram_debug.sent", symbol=symbol, decision=decision)
    except urllib.error.URLError as exc:
        log.error("telegram_publish_failed", symbol=symbol, err=str(exc))
