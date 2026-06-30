from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv
from shared.rabbitmq_config import resolve_rabbitmq_url

from src.amqp_safety import validate_rabbitmq_credentials
from src.binance_ws import BinanceKlineStream
from src.config import ServiceConfig
from src.publisher import LivePricePublisher

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = __import__("os").environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


async def _run() -> None:
    config = ServiceConfig.from_env()
    if not config.enabled:
        logger.info("BINANCE_LIVE_ENABLED=false — Binance live-price publisher idle")
        await asyncio.Event().wait()
        return

    symbols = ",".join(config.symbols)
    logger.info(
        "Binance live-price publisher starting symbols=%s timeframe=%s market=%s",
        symbols,
        config.timeframe,
        config.market,
    )

    rabbitmq_url = resolve_rabbitmq_url()
    user, vhost = validate_rabbitmq_credentials(rabbitmq_url)

    publisher = LivePricePublisher(rabbitmq_url, user=user, vhost=vhost)
    await publisher.connect()

    stream = BinanceKlineStream(config, publisher)
    try:
        await stream.run_forever()
    finally:
        await publisher.close()


def main() -> None:
    load_dotenv()
    _configure_logging()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Binance live-price publisher stopped")


if __name__ == "__main__":
    main()
