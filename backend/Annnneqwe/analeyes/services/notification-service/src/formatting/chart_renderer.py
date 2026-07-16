from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

logger = logging.getLogger(__name__)

CHART_WIDTH = 900
CHART_HEIGHT = 520
CHART_DPI = 100
MIN_CANDLES = 5
DEFAULT_MAX_CANDLES = 120
DEFAULT_MIN_CANDLES = 60
CHART_TMP_DIR = Path("/tmp/analeyes_charts")


class ChartRenderer:
    """Render dark candlestick charts from signal candle data — no LLM involvement."""

    def __init__(self, *, tmp_dir: Path | None = None) -> None:
        self._tmp_dir = tmp_dir or CHART_TMP_DIR

    def render(self, payload: dict[str, Any]) -> bytes | None:
        candles = self._extract_candles(payload)
        if not candles:
            logger.info(
                "signal_chart_skipped symbol=%s reason=no_candles",
                payload.get("symbol"),
            )
            return None

        rows = self._candles_to_rows(candles)
        if len(rows) < MIN_CANDLES:
            logger.info(
                "signal_chart_skipped symbol=%s reason=insufficient_candles count=%s",
                payload.get("symbol"),
                len(rows),
            )
            return None

        entry_price = self._entry_price(payload)
        try:
            frame = pd.DataFrame(rows).set_index("Date")
            image_bytes = self._plot_chart(frame, entry_price=entry_price, symbol=str(payload.get("symbol") or ""))
            if image_bytes:
                self._maybe_persist(symbol=str(payload.get("symbol") or "unknown"), image_bytes=image_bytes)
                logger.info(
                    "signal_chart_rendered symbol=%s candles=%s bytes=%s",
                    payload.get("symbol"),
                    len(rows),
                    len(image_bytes),
                )
            return image_bytes
        except Exception:
            logger.exception(
                "signal_chart_skipped symbol=%s reason=render_error",
                payload.get("symbol"),
            )
            return None

    def _extract_candles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw = payload.get("candles") or payload.get("historical_ohlcv") or []
        if not isinstance(raw, list):
            return []
        max_candles = int(os.environ.get("SIGNAL_CHART_MAX_CANDLES", str(DEFAULT_MAX_CANDLES)))
        min_take = min(DEFAULT_MIN_CANDLES, max_candles)
        if len(raw) > max_candles:
            return raw[-max_candles:]
        if len(raw) >= min_take:
            return raw
        return raw

    def _candles_to_rows(self, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            ts = (
                candle.get("timestamp")
                or candle.get("open_time")
                or candle.get("time")
                or candle.get("t")
                or candle.get("close_time")
            )
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)) or str(ts).isdigit():
                    ts_num = float(ts)
                    if ts_num > 1e12:
                        ts_num = ts_num / 1000.0
                    dt = pd.to_datetime(ts_num, unit="s", utc=True)
                else:
                    dt = pd.to_datetime(str(ts).replace("Z", "+00:00"), utc=True)
                rows.append(
                    {
                        "Date": dt,
                        "Open": float(candle.get("open") or candle.get("o") or 0),
                        "High": float(candle.get("high") or candle.get("h") or 0),
                        "Low": float(candle.get("low") or candle.get("l") or 0),
                        "Close": float(candle.get("close") or candle.get("c") or 0),
                        "Volume": float(candle.get("volume") or candle.get("v") or 0),
                    }
                )
            except (TypeError, ValueError):
                continue
        return rows

    @staticmethod
    def _entry_price(payload: dict[str, Any]) -> float | None:
        for key in ("entry_price", "entry", "current_price", "execution_price"):
            value = payload.get(key)
            if value is None or value == "" or str(value).lower() == "market":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _plot_chart(self, frame: pd.DataFrame, *, entry_price: float | None, symbol: str) -> bytes | None:
        mc = mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#0d1117",
            figcolor="#0d1117",
            edgecolor="#30363d",
            gridcolor="#21262d",
            gridstyle="--",
            rc={
                "axes.labelcolor": "#c9d1d9",
                "xtick.color": "#8b949e",
                "ytick.color": "#8b949e",
            },
        )

        addplot = None
        if entry_price is not None and not frame.empty:
            entry_line = pd.Series([entry_price] * len(frame), index=frame.index)
            addplot = mpf.make_addplot(
                entry_line,
                color="#f0b429",
                linestyle="--",
                width=1.2,
            )

        buf = io.BytesIO()
        figsize = (CHART_WIDTH / CHART_DPI, CHART_HEIGHT / CHART_DPI)
        has_volume = bool("Volume" in frame.columns and frame["Volume"].sum() > 0)
        kwargs: dict[str, Any] = {
            "type": "candle",
            "style": style,
            "volume": has_volume,
            "title": f"{symbol} — AnalEyes",
            "ylabel": "Price",
            "ylabel_lower": "Volume",
            "savefig": dict(fname=buf, dpi=CHART_DPI, bbox_inches="tight", facecolor="#0d1117"),
            "figsize": figsize,
            "datetime_format": "%H:%M",
            "xrotation": 0,
        }
        if addplot is not None:
            kwargs["addplot"] = addplot

        mpf.plot(frame, **kwargs)
        plt.close("all")
        buf.seek(0)
        data = buf.read()
        return data if data else None

    def _maybe_persist(self, *, symbol: str, image_bytes: bytes) -> None:
        try:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            safe_symbol = re.sub(r"[^A-Za-z0-9_-]", "_", symbol)
            path = self._tmp_dir / f"{safe_symbol}_latest.png"
            path.write_bytes(image_bytes)
        except OSError:
            logger.debug("signal_chart_persist_skipped symbol=%s", symbol)
