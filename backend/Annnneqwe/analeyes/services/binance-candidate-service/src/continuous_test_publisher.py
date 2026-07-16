"""DEV/TEST high-frequency publisher — disabled in production (ENABLE_HIGH_FREQUENCY_TEST_PARSER)."""

from __future__ import annotations

import asyncio
import logging
import os

from src.candle_buffer import CandleBuffer
from src.config import ServiceConfig
from src.converters.ae_brain_candidate import build_ae_brain_candidate
from src.publisher import CandidatePublisher

logger = logging.getLogger(__name__)


async def run_continuous_test_publisher(
    *,
    config: ServiceConfig,
    buffer: CandleBuffer,
    publisher: CandidatePublisher,
    stop_event: asyncio.Event,
) -> None:
    if not (config.enable_high_frequency_test_parser or config.continuous_test_mode):
        return

    interval_ms = int(os.environ.get("CANDIDATE_EMIT_INTERVAL_MS", "200"))
    interval_sec = interval_ms / 1000.0
    symbols = list(config.symbols)
    index = 0

    logger.warning(
        "high_frequency_test_parser enabled interval_ms=%s symbols=%s — NOT for production",
        interval_ms,
        len(symbols),
    )

    while not stop_event.is_set():
        if not symbols:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                continue
            break

        symbol = symbols[index % len(symbols)]
        index += 1

        latest = buffer.latest(symbol)
        candles = buffer.candles(symbol)
        if latest is None or len(candles) < config.window_candles:
            logger.info("high_frequency_test_skip symbol=%s reason=window_not_ready", symbol)
        else:
            closed = latest
            closed.closed = True
            payload = build_ae_brain_candidate(
                symbol=symbol,
                timeframe=config.timeframe,
                candles=candles,
                closed_candle=closed,
                market=config.market,
                window_candles=config.window_candles,
            )
            if payload is None:
                logger.info("high_frequency_test_skip symbol=%s reason=converter_returned_none", symbol)
            else:
                try:
                    await publisher.publish_candidate(payload)
                    logger.info("high_frequency_test_publish symbol=%s candles=%s", symbol, len(candles))
                except ValueError as exc:
                    logger.info("high_frequency_test_skip symbol=%s reason=%s", symbol, exc)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            break
        except asyncio.TimeoutError:
            continue
