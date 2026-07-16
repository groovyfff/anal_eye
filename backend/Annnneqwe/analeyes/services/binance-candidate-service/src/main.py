from __future__ import annotations

import asyncio
import contextlib
import logging
import sys

from dotenv import load_dotenv
from shared.binance_symbols import resolve_binance_symbols
from shared.rabbitmq_config import rabbitmq_connection_info, resolve_rabbitmq_url
from shared.symbol_universe import allowed_symbols_csv
from shared.utils.rabbitmq_topology import EXCHANGE, RoutingKey

from src.amqp_safety import validate_rabbitmq_credentials
from src.binance_ws import BinanceCandidateStream
from src.candle_buffer import CandleBuffer
from src.config import ServiceConfig
from src.continuous_test_publisher import run_continuous_test_publisher
from src.dedup_store import DedupStore
from src.publisher import CandidatePublisher
from src.rest_backfill import backfill_symbol, fetch_optional_market_fields
from src.rest_closed_candle_poller import RestClosedCandlePoller

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = __import__("os").environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


async def _backfill_all(config: ServiceConfig, buffer: CandleBuffer) -> dict[str, dict]:
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(6)
    optional_by_symbol: dict[str, dict] = {}

    async def _one(symbol: str) -> None:
        async with sem:
            await loop.run_in_executor(
                None,
                lambda sym=symbol: backfill_symbol(
                    symbol=sym,
                    interval=config.timeframe,
                    limit=config.window_candles,
                    rest_base_url=config.rest_base_url,
                    buffer=buffer,
                    publish_historical=config.backfill_publish_historical,
                ),
            )
            optional_by_symbol[symbol] = await loop.run_in_executor(
                None,
                lambda sym=symbol: fetch_optional_market_fields(
                    symbol=sym,
                    rest_base_url=config.rest_base_url,
                ),
            )

    results = await asyncio.gather(*[_one(sym) for sym in config.symbols], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        logger.error("REST backfill failures count=%s first=%s", len(failures), failures[0])
        raise failures[0]
    return optional_by_symbol


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

    if config.enable_legacy_parser:
        logger.warning("ENABLE_LEGACY_PARSER=true — legacy parser path is deprecated and disabled")
    if config.enable_high_frequency_test_parser:
        logger.warning("ENABLE_HIGH_FREQUENCY_TEST_PARSER=true — only for explicit dev testing")

    logger.info(
        "binance_candidate_startup allowed_symbols=%s subscribed_streams=%s timeframe=%s "
        "closed_candles_only=%s dedup_enabled=%s candidate_window_candles=%s "
        "exchange=%s routing_key=%s source=binance_futures_api",
        allowed_symbols_csv(),
        ",".join(config.all_stream_names()),
        config.timeframe,
        config.closed_candles_only,
        config.dedup_enabled,
        config.window_candles,
        EXCHANGE,
        RoutingKey.DATA_CANDIDATES_AI,
    )

    rabbitmq_url = resolve_rabbitmq_url()
    user, vhost = validate_rabbitmq_credentials(rabbitmq_url)
    rmq_info = rabbitmq_connection_info(rabbitmq_url)
    logger.info(
        "rabbitmq_publish_config exchange=%s routing_key=%s vhost=%s host=%s user=%s",
        EXCHANGE,
        RoutingKey.DATA_CANDIDATES_AI,
        rmq_info.get("vhost") or vhost,
        rmq_info.get("host") or "rabbitmq",
        user,
    )

    publisher = CandidatePublisher(rabbitmq_url, user=user, vhost=vhost)
    await publisher.connect()

    buffer = CandleBuffer(max_candles=config.max_candles)
    optional_features = await _backfill_all(config, buffer)
    dedup = DedupStore(config.dedup_db_path)

    stream = BinanceCandidateStream(
        config,
        buffer,
        publisher,
        dedup,
        optional_features=optional_features,
    )

    rest_poller: RestClosedCandlePoller | None = None
    if config.rest_fallback_enabled:
        rest_poller = RestClosedCandlePoller(
            config,
            buffer,
            stream,
            ws_health_getter=stream.ws_is_healthy,
        )
        logger.info(
            "rest_fallback_enabled=true poll_sec=%s always_on=%s ws_idle_timeout_sec=%s",
            config.rest_fallback_poll_sec,
            config.rest_fallback_always_on,
            config.ws_idle_timeout_sec,
        )
    else:
        logger.info("rest_fallback_enabled=false — WebSocket is the only source")

    stop_event = asyncio.Event()
    continuous_task: asyncio.Task[None] | None = None
    if config.enable_high_frequency_test_parser or config.continuous_test_mode:
        logger.warning(
            "continuous_test_publisher_enabled high_frequency=%s continuous_test_mode=%s "
            "(dev/test only — never enabled in prod/staging)",
            config.enable_high_frequency_test_parser,
            config.continuous_test_mode,
        )
        continuous_task = asyncio.create_task(
            run_continuous_test_publisher(
                config=config,
                buffer=buffer,
                publisher=publisher,
                stop_event=stop_event,
            ),
            name="high-frequency-test-publisher",
        )
    try:
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(stream.run_forever(), name="binance-ws-stream"),
        ]
        if rest_poller is not None:
            tasks.append(
                asyncio.create_task(rest_poller.run_forever(), name="rest-closed-candle-poller")
            )
        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        if continuous_task is not None:
            continuous_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await continuous_task
        dedup.close()
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
