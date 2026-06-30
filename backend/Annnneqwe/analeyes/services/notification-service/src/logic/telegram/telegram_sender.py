from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

logger = logging.getLogger(__name__)


class TelegramDeliveryError(RuntimeError):
    """Raised when Telegram delivery fails after retries."""


@dataclass
class SendResult:
    sent: bool
    skip_reason: str | None = None


def markdown_protect(text: str) -> str:
    """Escape characters that break Telegram Markdown legacy mode."""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))


def _format_price(price: Any) -> str:
    if price is None:
        return "-"
    try:
        value = float(price)
        return f"{value:.6g}"
    except (TypeError, ValueError):
        return str(price)


class TelegramSender:
    """Format and deliver notifications via the Telegram Bot API."""

    def __init__(self, config: dict[str, Any]) -> None:
        telegram_cfg = config.get("telegram", {}) or {}
        env_group = os.environ.get("TELEGRAM_GROUP_ID", "").strip()
        self.group_id = int(env_group) if env_group else telegram_cfg.get("group_id")
        self.asset_class_topics = telegram_cfg.get("asset_class_topics", {}) or {}
        self.ai_specific_topics = telegram_cfg.get("ai_specific_topics", {}) or {}
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.admin_id = os.environ.get("TELEGRAM_ADMIN_ID", "").strip()
        self.retry_attempts = int(telegram_cfg.get("retry_attempts", 3))
        self.allow_no_topic = os.environ.get("TELEGRAM_ALLOW_NO_TOPIC", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    @staticmethod
    def normalize_signal_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Map alternate field names from producers to what the sender expects."""
        normalized = dict(payload)
        decision = payload.get("decision") or payload.get("signal_type") or payload.get("side")
        if decision is not None:
            normalized["decision"] = decision
        if normalized.get("reason") is None and payload.get("reason_summary") is not None:
            normalized["reason"] = payload.get("reason_summary")
        if normalized.get("tp") is None and payload.get("tp_price") is not None:
            normalized["tp"] = payload.get("tp_price")
        if normalized.get("sl") is None and payload.get("sl_price") is not None:
            normalized["sl"] = payload.get("sl_price")
        if normalized.get("consensus_achieved") is None and payload.get("ai_consensus_achieved") is not None:
            normalized["consensus_achieved"] = payload.get("ai_consensus_achieved")
        return normalized

    def resolve_topic_id(self, asset_class: str | None, source_ai: str | None) -> int | None:
        source = (source_ai or "").strip().lower()
        if source and source in self.ai_specific_topics:
            topic = self.ai_specific_topics.get(source)
            return int(topic) if topic is not None else None
        if asset_class and asset_class in self.asset_class_topics:
            topic = self.asset_class_topics.get(asset_class)
            return int(topic) if topic is not None else None
        return None

    def _skip_reason(self, source_ai: str | None, topic_id: int | None) -> str | None:
        if not self.bot_token:
            return "TELEGRAM_BOT_TOKEN is not set"
        if not self.group_id:
            return "telegram group_id / TELEGRAM_GROUP_ID is not configured"
        if topic_id is None and not self.allow_no_topic:
            return f"no forum topic mapping for source_ai={source_ai!r} (set TELEGRAM_ALLOW_NO_TOPIC=true to send without topic)"
        return None

    def _build_message_payload(self, text: str, topic_id: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": self.group_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        if topic_id is not None:
            payload["message_thread_id"] = topic_id
        return payload

    async def _post_telegram(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if not data.get("ok"):
                        desc = data.get("description", data)
                        migrated = re.search(r"new chat id:\s*(-?\d+)", str(desc), re.I)
                        if migrated and method == "sendMessage":
                            payload["chat_id"] = int(migrated.group(1))
                            logger.warning("Telegram group migrated; retrying with chat_id=%s", payload["chat_id"])
                            continue
                        logger.error("Telegram API %s rejected: %s", method, desc)
                        raise TelegramDeliveryError(f"Telegram API {method} rejected: {desc}")
                    return data
            except TelegramDeliveryError:
                raise
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                body = exc.response.text[:500] if exc.response is not None else ""
                logger.warning(
                    "Telegram %s HTTP %s (attempt %d/%d) body=%s",
                    method,
                    exc.response.status_code if exc.response is not None else "?",
                    attempt,
                    self.retry_attempts,
                    body,
                )
                if exc.response is not None and exc.response.status_code == 400 and "parse" in body.lower():
                    plain = dict(payload)
                    plain.pop("parse_mode", None)
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.post(url, json=plain)
                            response.raise_for_status()
                            data = response.json()
                            if data.get("ok"):
                                logger.info("Telegram %s sent with plain-text fallback", method)
                                return data
                    except Exception:
                        logger.exception("Telegram plain-text fallback failed for %s", method)
            except Exception as exc:
                last_exc = exc
                logger.exception("Telegram %s failed (attempt %d/%d)", method, attempt, self.retry_attempts)

            if attempt < self.retry_attempts:
                await asyncio.sleep(min(2 ** attempt, 10))

        logger.exception("Telegram %s failed after %d attempts", method, self.retry_attempts)
        raise TelegramDeliveryError(f"Telegram {method} failed after {self.retry_attempts} attempts") from last_exc

    def _format_signal_message(self, payload: dict[str, Any]) -> str:
        asset_class = str(payload.get("asset_class", "crypto"))
        features = payload.get("features") or {}
        lines = [
            f"🚀 *{payload.get('name') or payload.get('symbol')}* ({asset_class})",
            f"Direction: *{payload.get('decision')}* | Confidence: {payload.get('confidence')}",
            f"Entry: {_format_price(payload.get('entry_price'))} | TP: {_format_price(payload.get('tp'))} | SL: {_format_price(payload.get('sl'))}",
        ]
        if asset_class == "crypto":
            for key in ("funding_rate", "open_interest_z", "cvd"):
                if features.get(key) is not None:
                    lines.append(f"{key}: {features[key]}")
        else:
            for key in ("rsi", "macd_hist", "vol_rel"):
                if features.get(key) is not None:
                    lines.append(f"{key}: {features[key]}")
        reason = payload.get("reason")
        if reason:
            lines.append(f"Reason: {reason}")
        return "\n".join(lines)

    def _generate_signal_chart_image(self, payload: dict[str, Any]) -> bytes | None:
        asset_class = str(payload.get("asset_class", "crypto"))
        ohlcv = payload.get("historical_ohlcv") or []
        if asset_class == "crypto" or not ohlcv:
            return None
        rows = []
        for candle in ohlcv:
            ts = candle.get("timestamp")
            if not ts:
                continue
            rows.append(
                {
                    "Date": pd.to_datetime(str(ts).replace("Z", "+00:00")),
                    "Open": candle.get("open"),
                    "High": candle.get("high"),
                    "Low": candle.get("low"),
                    "Close": candle.get("close"),
                    "Volume": candle.get("volume") or 0,
                }
            )
        if len(rows) < 5:
            return None
        try:
            frame = pd.DataFrame(rows).set_index("Date")
            buf = io.BytesIO()
            mpf.plot(
                frame,
                type="candle",
                style="charles",
                volume=True,
                savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
            )
            plt.close("all")
            buf.seek(0)
            return buf.read()
        except Exception:
            logger.exception("Chart generation disabled due to error for symbol=%s", payload.get("symbol"))
            return None

    async def send_signal(self, payload: dict[str, Any]) -> SendResult:
        payload = self.normalize_signal_payload(payload)
        source_ai = payload.get("source_ai")
        symbol = payload.get("symbol")
        decision = payload.get("decision")
        topic_id = self.resolve_topic_id(payload.get("asset_class"), source_ai)

        skip = self._skip_reason(source_ai, topic_id)
        if skip:
            logger.info(
                "Telegram signal skipped symbol=%s decision=%s source_ai=%s reason=%s",
                symbol,
                decision,
                source_ai,
                skip,
            )
            return SendResult(sent=False, skip_reason=skip)

        text = self._format_signal_message(payload)
        chart = self._generate_signal_chart_image(payload)
        if chart is None and payload.get("historical_ohlcv"):
            logger.info(
                "Chart generation skipped/disabled for symbol=%s (optional)",
                symbol,
            )

        try:
            await self._post_telegram("sendMessage", self._build_message_payload(text, topic_id))
        except TelegramDeliveryError:
            logger.exception(
                "Telegram API failed for signal symbol=%s decision=%s source_ai=%s",
                symbol,
                decision,
                source_ai,
            )
            raise
        except Exception:
            logger.exception(
                "Telegram send failed for signal symbol=%s decision=%s source_ai=%s",
                symbol,
                decision,
                source_ai,
            )
            raise

        logger.info(
            "Telegram signal sent symbol=%s topic_id=%s chart=%s",
            symbol,
            topic_id,
            bool(chart),
        )
        return SendResult(sent=True)

    async def send_signal_outcome(self, payload: dict[str, Any]) -> SendResult:
        source_ai = payload.get("source_ai")
        topic_id = self.resolve_topic_id(payload.get("asset_class"), source_ai)
        skip = self._skip_reason(source_ai, topic_id)
        if skip:
            logger.info(
                "Telegram outcome skipped symbol=%s status=%s reason=%s",
                payload.get("symbol"),
                payload.get("status"),
                skip,
            )
            return SendResult(sent=False, skip_reason=skip)

        status = str(payload.get("status", "unknown")).upper()
        emoji = {"TP": "🎯", "SL": "🛑", "EXPIRED": "⏰", "CANCELLED": "❌"}.get(status, "📊")
        topic_id = self.resolve_topic_id(payload.get("asset_class"), source_ai)
        text = (
            f"{emoji} *Outcome* {payload.get('symbol')}\n"
            f"Status: *{status}*\n"
            f"PnL: {payload.get('pnl_percent')}%"
        )
        await self._post_telegram("sendMessage", self._build_message_payload(text, topic_id))
        return SendResult(sent=True)

    async def send_entry_event_notification(self, payload: dict[str, Any]) -> SendResult:
        source_ai = payload.get("source_ai")
        topic_id = self.resolve_topic_id(payload.get("asset_class"), source_ai)
        skip = self._skip_reason(source_ai, topic_id)
        if skip:
            logger.info(
                "Telegram entry skipped symbol=%s reason=%s",
                payload.get("symbol"),
                skip,
            )
            return SendResult(sent=False, skip_reason=skip)

        topic_id = self.resolve_topic_id(payload.get("asset_class"), source_ai)
        text = (
            f"✅ *Entry* {payload.get('symbol')}\n"
            f"Fill price: {_format_price(payload.get('fill_price'))}"
        )
        await self._post_telegram("sendMessage", self._build_message_payload(text, topic_id))
        return SendResult(sent=True)

    # Back-compat aliases used by main.py
    async def send_outcome(self, payload: dict[str, Any]) -> SendResult:
        return await self.send_signal_outcome(payload)

    async def send_entry_event(self, payload: dict[str, Any]) -> SendResult:
        return await self.send_entry_event_notification(payload)
