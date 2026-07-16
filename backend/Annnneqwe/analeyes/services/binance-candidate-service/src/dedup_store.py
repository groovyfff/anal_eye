"""Persistent per-symbol/timeframe/candle deduplication store."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(os.environ.get("CANDIDATE_DEDUP_DB_PATH", "/app/data/candidate_dedup.db"))


class DedupStore:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path or _DEFAULT_DB)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS published_candles (
                dedup_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle_open_time INTEGER NOT NULL,
                candle_close_time INTEGER NOT NULL,
                published_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(symbol: str, timeframe: str, candle_open_time: int) -> str:
        return f"{symbol.upper()}:{timeframe}:{candle_open_time}"

    def was_published(self, symbol: str, timeframe: str, candle_open_time: int) -> bool:
        key = self.make_key(symbol, timeframe, candle_open_time)
        row = self._conn.execute(
            "SELECT 1 FROM published_candles WHERE dedup_key = ?",
            (key,),
        ).fetchone()
        return row is not None

    def mark_published(
        self,
        symbol: str,
        timeframe: str,
        candle_open_time: int,
        candle_close_time: int,
    ) -> None:
        key = self.make_key(symbol, timeframe, candle_open_time)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO published_candles
            (dedup_key, symbol, timeframe, candle_open_time, candle_close_time, published_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, symbol.upper(), timeframe, candle_open_time, candle_close_time, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
