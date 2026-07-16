"""Backward-compatible re-export — use rest_backfill.fetch_klines."""

from __future__ import annotations

from src.rest_backfill import fetch_klines as fetch_bootstrap_klines

__all__ = ["fetch_bootstrap_klines"]
