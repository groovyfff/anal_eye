from __future__ import annotations

import asyncio
import contextlib
import logging
import sys

from dotenv import load_dotenv
from shared.binance_symbols import resolve_binance_symbols
from shared.rabbitmq_config import resolve_rabbitmq_url

from src.amqp_safety import validate_rabbitmq_credentials
from src.binance_ws import BinanceCandidateStream
from src.bootstrap import fetch_bootstrap_klines
from src.candle_buffer import CandleBuffer
from src.config import ServiceConfig
from src.continuous_test_publisher import run_continuous_test_publisher
from src.publish_policy import PublishPolicy
from src.publisher import CandidatePublisher

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = __import__("os").environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


async def _bootstrap_symbols(config: ServiceConfig, buffer: CandleBuffer) -> None:
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(10)

    async def _bootstrap_one(symbol: str) -> None:
        async with sem:
            candles = await loop.run_in_executor(
                None,
                lambda sym=symbol: fetch_bootstrap_klines(
                    symbol=sym,
                    interval=config.timeframe,
                    limit=config.bootstrap_limit,
                    rest_base_url=config.rest_base_url,
                ),
            )
            count = buffer.load_bootstrap(symbol, candles)
            logger.info(
                "Bootstrapped Binance candles symbol=%s timeframe=%s count=%s",
                symbol,
                config.timeframe,
                count,
            )

    results = await asyncio.gather(*[_bootstrap_one(symbol) for symbol in config.symbols], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        logger.error("Bootstrap failures count=%s first=%s", len(failures), failures[0])
        raise failures[0]


async def _run() -> None:
    loop = asyncio.get_running_loop()
    symbols = await loop.run_in_executor(
        None,
        lambda: resolve_binance_symbols(
            rest_base_url=(__import__("os").environ.get("BINANCE_REST_BASE_URL") or "https://fapi.binance.com"),
        ),
    )
    config = ServiceConfig.from_env(symbols=symbols)
    if not config.enabled:
        logger.info("BINANCE_CANDIDATE_ENABLED=false — Binance candidate publisher idle")
        await asyncio.Event().wait()
        return

    symbols = ",".join(config.symbols)
    logger.info(
        "Binance candidate publisher starting symbols=%s timeframe=%s min_candles=%s mode=%s",
        symbols,
        config.timeframe,
        config.min_candles,
        config.publish_mode,
    )

    rabbitmq_url = resolve_rabbitmq_url()
    user, vhost = validate_rabbitmq_credentials(rabbitmq_url)

    publisher = CandidatePublisher(rabbitmq_url, user=user, vhost=vhost)
    await publisher.connect()

    buffer = CandleBuffer(max_candles=config.max_candles)
    await _bootstrap_symbols(config, buffer)

    policy = PublishPolicy(
        throttle_sec=config.throttle_sec,
        publish_on_candle_close=config.publish_on_candle_close,
        publish_on_every_update=config.publish_on_every_update,
    )
    stream = BinanceCandidateStream(config, buffer, publisher, policy)
    stop_event = asyncio.Event()
    continuous_task: asyncio.Task[None] | None = None
    if config.continuous_test_mode:
        continuous_task = asyncio.create_task(
            run_continuous_test_publisher(
                config=config,
                buffer=buffer,
                publisher=publisher,
                stop_event=stop_event,
            ),
            name="continuous-test-publisher",
        )
    try:
        await stream.run_forever()
    finally:
        stop_event.set()
        if continuous_task is not None:
            continuous_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await continuous_task
        await publisher.close()


def main() -> None:
    load_dotenv()
    _configure_logging()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Binance candidate publisher stopped")


if __name__ == "__main__":
    main()
