"""Backward-compatible import path."""
from src.logic.telegram.telegram_sender import SendResult, TelegramDeliveryError, TelegramSender

__all__ = ["SendResult", "TelegramDeliveryError", "TelegramSender"]
