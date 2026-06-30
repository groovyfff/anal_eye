"""DEV/TEST continuous candidate publisher (round-robin, in-memory buffers only)."""

from __future__ import annotations

import asyncio
import logging

from src.candle_buffer import CandleBuffer
from src.candidate_builder import build_candidate_payload
from src.config import ServiceConfig
from src.publisher import CandidatePublisher

logger = logging.getLogger(__name__)


async def run_continuous_test_publisher(
    *,
    config: ServiceConfig,
    buffer: CandleBuffer,
    publisher: CandidatePublisher,
    stop_event: asyncio.Event,
) -> None:
    if not config.continuous_test_mode:
        return

    interval_sec = config.emit_interval_ms / 1000.0
    symbols = list(config.symbols)
    index = 0

    logger.info(
        "continuous_test_mode enabled interval_ms=%s symbols=%s round_robin=%s require_min_candles=%s",
        config.emit_interval_ms,
        len(symbols),
        config.emit_round_robin,
        config.emit_require_min_candles,
    )

    while not stop_event.is_set():
        if not symbols:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                continue
            break

        symbol = symbols[index % len(symbols)]
        if config.emit_round_robin:
            index += 1

        count = buffer.count(symbol)
        if config.emit_require_min_candles and count < config.min_candles:
            logger.info(
                "continuous_test_skip symbol=%s reason=not_enough_candles count=%s min=%s",
                symbol,
                count,
                config.min_candles,
            )
        elif count == 0:
            logger.info("continuous_test_skip symbol=%s reason=missing_buffer", symbol)
        else:
            latest = buffer.latest(symbol)
            candles = buffer.candles(symbol)
            if latest is None:
                logger.info("continuous_test_skip symbol=%s reason=missing_buffer", symbol)
            else:
                try:
                    payload = build_candidate_payload(
                        symbol=symbol,
                        market=config.market,
                        timeframe=config.timeframe,
                        event_time=latest.timestamp,
                        candles=candles,
                    )
                    await publisher.publish_candidate(payload)
                    logger.info(
                        "continuous_test_publish symbol=%s interval_ms=%s candles=%s",
                        symbol,
                        config.emit_interval_ms,
                        len(candles),
                    )
                except ValueError as exc:
                    logger.info("continuous_test_skip symbol=%s reason=%s", symbol, exc)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            break
        except asyncio.TimeoutError:
            continue
