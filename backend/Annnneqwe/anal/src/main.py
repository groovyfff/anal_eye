from __future__ import annotations
import asyncio
import contextlib
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar
from src.collectors.base_collector import BaseCollector, Quote
from src.collectors.polygon_collector import PolygonCollector
from src.collectors.yahoo_finance_collector import YahooFinanceCollector
from src.logic.composite_score import CompositeScore
from src.logic.feature_generator import FeatureGenerator
from src.logic.market_hours import MarketHours
from src.logic.pattern_engine import PatternEngine
from src.logic.payload_validator import validate_candidate_payload, validate_live_payload
from src.logic.trigger_engine import TriggerEngine, TriggerResult
from src.settings import get_watchlist_items, load_settings
from src.utils.logger import get_logger, setup_logging
from src.utils.pika_client import PikaClient
_T = TypeVar('_T')

class ExternalMarketsService:

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        setup_logging(settings.get('logging', {}))
        self.logger = get_logger(__name__)
        self.exchange_name = settings.get('rabbitmq', {}).get('exchange', 'analeyes_exchange')
        self.candidate_routing_key = 'data.candidates.ai'
        self.price_routing_key = 'data.live_prices.external'
        self.main_timeframe = settings.get('main_timeframe', '5m')
        self.context_timeframes = settings.get('context_timeframes', ['1h', '4h'])
        self.history_depth = self._clamp_int_setting(raw_value=settings.get('history_depth', 200), min_value=2, max_value=200, setting_name='history_depth')
        self.scan_interval_s = int(settings.get('scan_interval_s', 60))
        self.price_broadcast_interval_s = int(settings.get('price_broadcast_interval_s', 5))
        self.quote_freshness_ms = int(settings.get('quote_freshness_ms', 4500))
        self.price_fetch_workers = max(1, int(settings.get('price_fetch_workers', 8)))
        self.scan_workers = self._clamp_int_setting(raw_value=settings.get('scan_workers', 4), min_value=1, max_value=64, setting_name='scan_workers')
        self.use_thread_pool = bool(settings.get('use_thread_pool', False))
        self._thread_pool_fallback_warned = False
        self.market_hours = MarketHours(settings.get('market_hours', {}))
        trigger_cfg = settings.get('triggers', {})
        ema_cfg = trigger_cfg.get('ema_crossover', {})
        feature_cfg = {'breakout_lookback': trigger_cfg.get('price_breakout', {}).get('lookback_periods', 20), 'volume_window': trigger_cfg.get('volume_spike', {}).get('window', 20), 'ema_fast_period': ema_cfg.get('fast_period', 8), 'ema_slow_period': ema_cfg.get('slow_period', 21)}
        self.feature_generator = FeatureGenerator(feature_cfg)
        self.trigger_engine = TriggerEngine(trigger_cfg)
        self.composite_score = CompositeScore(settings.get('composite_score', {}))
        self.pattern_engine = PatternEngine(settings.get('patterns', {}))
        self.watchlist_items = get_watchlist_items(settings)
        self.scan_candidates = [item for item in self.watchlist_items if item['asset_class'] in {'stock', 'metal', 'forex'} and (not item.get('use_for_correlation', False))]
        self.broadcast_items = [item for item in self.watchlist_items if item['asset_class'] in {'stock', 'metal', 'forex'} and (not item.get('use_for_correlation', False))]
        self._has_stock_candidates = any((item['asset_class'] == 'stock' for item in self.scan_candidates))
        self._has_metal_candidates = any((item['asset_class'] == 'metal' for item in self.scan_candidates))
        self.reference_sp500 = self._find_reference_symbol({'^GSPC', 'SPY'})
        if self.reference_sp500 is None and self._has_stock_candidates:
            self.reference_sp500 = 'SPY'
            self.logger.warning('Missing S&P500 benchmark in watchlist, using fallback symbol SPY', extra={'symbol': '-'})
        self.reference_dxy = self._find_reference_symbol({'DX-Y.NYB', 'DXY'})
        if self.reference_dxy is None and self._has_metal_candidates:
            self.reference_dxy = 'DX-Y.NYB'
            self.logger.warning('Missing DXY benchmark in watchlist, using fallback symbol DX-Y.NYB', extra={'symbol': '-'})
        provider_cfg = settings.get('data_provider', {})
        self.collector = self._build_collector(provider_cfg)
        rabbit_cfg = settings.get('rabbitmq', {})
        self.rabbit_connect_retries = int(rabbit_cfg.get('connect_retries', 10))
        self.rabbit_connect_retry_delay_s = float(rabbit_cfg.get('connect_retry_delay_s', 2.0))
        candidate_queue_name = str(rabbit_cfg.get('candidate_queue', self.candidate_routing_key))
        live_queue_name = str(rabbit_cfg.get('live_prices_queue', self.price_routing_key))
        self.rabbit = PikaClient(url=rabbit_cfg.get('url', 'amqp://user:password@rabbitmq:5672/'), default_exchange=self.exchange_name, queue_bindings=[(candidate_queue_name, self.candidate_routing_key), (live_queue_name, self.price_routing_key)])
        self._quote_cache: dict[str, Quote] = {}
        self._last_published_candidate_timestamps: dict[str, str] = {}
        self._missing_asset_feature_warnings: set[tuple[str, str]] = set()
        self._shutdown_event = asyncio.Event()

    def _build_collector(self, provider_cfg: dict[str, Any]) -> BaseCollector:
        provider = str(provider_cfg.get('name', 'yahoo_finance')).strip().lower()
        api_key = provider_cfg.get('api_key', '')
        request_delay_s = float(provider_cfg.get('request_delay_s', 0.25))
        raw_quote_delay = provider_cfg.get('quote_request_delay_s')
        quote_request_delay_s: float | None
        if raw_quote_delay is None:
            quote_request_delay_s = None
        else:
            quote_request_delay_s = float(raw_quote_delay)
        max_retries = self._clamp_int_setting(raw_value=provider_cfg.get('max_retries', 5), min_value=1, max_value=5, setting_name='data_provider.max_retries')
        market_hours_cfg = self.settings.get('market_hours', {})
        include_prepost = bool(market_hours_cfg.get('pre_market_enabled', False) or market_hours_cfg.get('after_hours_enabled', False))
        if provider == 'polygon':
            return PolygonCollector(api_key=api_key, request_delay_s=request_delay_s, quote_request_delay_s=quote_request_delay_s, max_retries=max_retries)
        if provider in {'yahoo_finance', 'yahoo'}:
            return YahooFinanceCollector(api_key=api_key, request_delay_s=request_delay_s, quote_request_delay_s=quote_request_delay_s, max_retries=max_retries, include_prepost=include_prepost)
        supported = ['yahoo_finance', 'polygon']
        raise ValueError(f"Unsupported data_provider.name '{provider}'. Supported: {', '.join(supported)}")

    def _find_reference_symbol(self, candidates: set[str]) -> str | None:
        for item in self.watchlist_items:
            symbol = item['symbol']
            if symbol in candidates:
                return symbol
        return None

    async def run(self) -> None:
        connected = False
        for attempt in range(1, self.rabbit_connect_retries + 1):
            connected = await self.rabbit.connect()
            if connected:
                break
            self.logger.warning('RabbitMQ unavailable, retry %s/%s in %.1fs', attempt, self.rabbit_connect_retries, self.rabbit_connect_retry_delay_s, extra={'symbol': '-'})
            if attempt < self.rabbit_connect_retries:
                await asyncio.sleep(self.rabbit_connect_retry_delay_s)
        if not connected:
            raise RuntimeError(f'Failed to connect to RabbitMQ after {self.rabbit_connect_retries} attempts')
        self.logger.info('Service started with %s scan symbols and %s broadcast symbols', len(self.scan_candidates), len(self.broadcast_items), extra={'symbol': '-'})
        scan_task = asyncio.create_task(self._scan_loop(), name='scan_loop')
        broadcast_task = asyncio.create_task(self._broadcast_loop(), name='broadcast_loop')
        await self._shutdown_event.wait()
        for task in (scan_task, broadcast_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.rabbit.close()

    async def stop(self) -> None:
        self._shutdown_event.set()

    async def _scan_loop(self) -> None:
        while not self._shutdown_event.is_set():
            started = asyncio.get_running_loop().time()
            try:
                await self._scan_once()
            except Exception as exc:
                self.logger.error('Unhandled scan loop error: %s', exc, extra={'symbol': '-'})
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, self.scan_interval_s - elapsed))

    async def _scan_once(self) -> None:
        now_utc = dt.datetime.now(tz=dt.timezone.utc)
        benchmark_close = None
        dxy_close = None
        stock_session_open = self.market_hours.is_market_open('stock', now_utc=now_utc)
        metal_session_open = self.market_hours.is_market_open('metal', now_utc=now_utc)
        try:
            if self.reference_sp500 and stock_session_open and self._has_stock_candidates:
                benchmark_close = await self._fetch_close_series(symbol=self.reference_sp500, timeframe=self.main_timeframe, limit=self.history_depth)
        except Exception as exc:
            self.logger.error('Failed to load S&P500 benchmark: %s', exc, extra={'symbol': '-'})
        try:
            if self.reference_dxy and metal_session_open and self._has_metal_candidates:
                dxy_close = await self._fetch_close_series(symbol=self.reference_dxy, timeframe=self.main_timeframe, limit=self.history_depth)
        except Exception as exc:
            self.logger.error('Failed to load DXY benchmark: %s', exc, extra={'symbol': '-'})
        semaphore = asyncio.Semaphore(self.scan_workers)

        async def _scan_item(item: dict[str, Any]) -> None:
            async with semaphore:
                await self._scan_symbol(item=item, now_utc=now_utc, benchmark_close=benchmark_close, dxy_close=dxy_close)
        await asyncio.gather(*(_scan_item(item) for item in self.scan_candidates))

    async def _scan_symbol(self, item: dict[str, Any], now_utc: dt.datetime, benchmark_close: Any, dxy_close: Any) -> None:
        symbol = item['symbol']
        asset_class = item['asset_class']
        if not self.market_hours.is_market_open(asset_class=asset_class, now_utc=now_utc):
            return
        try:
            candle_frame = await self._fetch_ohlcv(symbol=symbol, timeframe=self.main_timeframe, limit=self.history_depth)
            quote = await self._fetch_quote_with_cache(symbol=symbol, now_utc=now_utc, allow_stale=False)
            if quote is None:
                return
            context_trends = await self._build_context_trends(symbol=symbol, asset_class=asset_class, now_utc=now_utc)
            indicator_df = self.feature_generator.compute_indicators(frame=candle_frame, asset_class=asset_class, benchmark_close=benchmark_close if asset_class == 'stock' else None, dxy_close=dxy_close if asset_class == 'metal' else None)
            indicator_df.attrs['symbol'] = symbol
            analysis_df = self._closed_candle_view(indicator_df=indicator_df, now_utc=now_utc, timeframe=self.main_timeframe)
            if len(analysis_df) < 2:
                return
            features = self.feature_generator.build_feature_payload(indicator_df=analysis_df, asset_class=asset_class, bid=quote.bid, ask=quote.ask)
            features['current_price'] = quote.price
            features.update(self._context_feature_payload(context_trends))
            if not self._has_required_asset_specific_features(symbol=symbol, asset_class=asset_class, features=features):
                return
            trigger_result = self.trigger_engine.evaluate(analysis_df, context_trends=context_trends)
            if not trigger_result.triggered:
                return
            pattern_detection = self.pattern_engine.analyze(analysis_df)
            score = self.composite_score.calculate(features=features, pattern_score=pattern_detection.pattern_score)
            if not self.composite_score.should_publish(score):
                return
            signal_timestamp = self._signal_timestamp_from_closed_candle(indicator_df=analysis_df, timeframe=self.main_timeframe)
            if signal_timestamp is None:
                self.logger.warning('Skipping candidate: unable to determine closed-candle timestamp', extra={'symbol': symbol})
                return
            dedup_key = self._candidate_dedup_key(signal_timestamp=signal_timestamp)
            if self._last_published_candidate_timestamps.get(symbol) == dedup_key:
                self.logger.debug('Skipping duplicate candidate for closed candle', extra={'symbol': symbol})
                return
            payload = self._build_candidate_payload(symbol=symbol, asset_class=asset_class, features=features, trigger_result=trigger_result, indicator_df=analysis_df, patterns_payload=pattern_detection.patterns_payload, composite_score=score, signal_timestamp=signal_timestamp)
            published = await self._publish_json(exchange_name=self.exchange_name, routing_key=self.candidate_routing_key, payload=payload)
            if published is False:
                self.logger.warning('Candidate publish rejected by payload validator', extra={'symbol': symbol})
                return
            self._last_published_candidate_timestamps[symbol] = dedup_key
            self.logger.info('Published candidate (%s)', ','.join(trigger_result.reasons), extra={'symbol': symbol})
        except Exception as exc:
            self.logger.error('Scan failed: %s', exc, extra={'symbol': symbol})

    async def _build_context_trends(self, symbol: str, asset_class: str, now_utc: dt.datetime) -> dict[str, dict[str, Any]]:
        trends: dict[str, dict[str, Any]] = {}

        async def _load_timeframe(timeframe: str) -> tuple[str, dict[str, Any] | None]:
            try:
                frame = await self._fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=self.history_depth)
                context_df = self.feature_generator.compute_indicators(frame=frame, asset_class=asset_class)
                closed_context_df = self._closed_candle_view(indicator_df=context_df, now_utc=now_utc, timeframe=timeframe)
                if closed_context_df.empty:
                    return (timeframe, None)
                return (timeframe, self.feature_generator.build_trend_context(closed_context_df))
            except Exception as exc:
                self.logger.error('Context timeframe fetch failed (%s): %s', timeframe, exc, extra={'symbol': symbol})
                return (timeframe, None)
        results = await asyncio.gather(*(_load_timeframe(timeframe) for timeframe in self.context_timeframes))
        for timeframe, trend in results:
            if trend is not None:
                trends[timeframe] = trend
        return trends

    def _context_feature_payload(self, context_trends: dict[str, dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for timeframe, trend in context_trends.items():
            direction = str(trend.get('direction', 'NEUTRAL')).upper()
            strength = self._safe_strength(trend.get('strength'))
            bias = 0.0
            if direction == 'LONG':
                bias = 1.0
            elif direction == 'SHORT':
                bias = -1.0
            payload[f'trend_context_{timeframe}_direction'] = direction
            payload[f'trend_context_{timeframe}_strength'] = strength
            payload[f'trend_context_{timeframe}_bias'] = bias
        return payload

    def _has_required_asset_specific_features(self, symbol: str, asset_class: str, features: dict[str, Any]) -> bool:
        if asset_class == 'stock':
            if features.get('sp500_correlation') is None:
                self._warn_missing_asset_feature_once(symbol=symbol, feature_name='sp500_correlation')
        elif asset_class == 'metal':
            if features.get('dxy_correlation') is None:
                self._warn_missing_asset_feature_once(symbol=symbol, feature_name='dxy_correlation')
        elif asset_class == 'forex':
            if features.get('bid_ask_spread_pips') is None:
                self._warn_missing_asset_feature_once(symbol=symbol, feature_name='bid_ask_spread_pips')
        return True

    def _warn_missing_asset_feature_once(self, symbol: str, feature_name: str) -> None:
        warning_key = (symbol, feature_name)
        if warning_key in self._missing_asset_feature_warnings:
            return
        self._missing_asset_feature_warnings.add(warning_key)
        self.logger.warning('Missing asset-specific feature %s; publishing candidate with null value', feature_name, extra={'symbol': symbol})

    @staticmethod
    def _safe_strength(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric != numeric:
            return 0.0
        return round(max(0.0, min(numeric, 1.0)), 4)

    async def _broadcast_loop(self) -> None:
        while not self._shutdown_event.is_set():
            started = asyncio.get_running_loop().time()
            try:
                await self._broadcast_once()
            except Exception as exc:
                self.logger.error('Unhandled broadcast loop error: %s', exc, extra={'symbol': '-'})
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, self.price_broadcast_interval_s - elapsed))

    async def _broadcast_once(self) -> None:
        now_utc = dt.datetime.now(tz=dt.timezone.utc)
        active_items = [item for item in self.broadcast_items if self.market_hours.is_market_open(asset_class=item['asset_class'], now_utc=now_utc)]
        if not active_items:
            return
        stale_symbols = [item['symbol'] for item in active_items if not self._is_quote_fresh(self._quote_cache.get(item['symbol']), now_utc)]
        await self._refresh_quotes(stale_symbols)
        for item in active_items:
            symbol = item['symbol']
            asset_class = item['asset_class']
            quote = self._quote_cache.get(symbol)
            pre_publish_now_utc = dt.datetime.now(tz=dt.timezone.utc)
            if not self._is_quote_fresh(quote, pre_publish_now_utc):
                try:
                    quote = await self._fetch_quote(symbol=symbol)
                except Exception as exc:
                    self.logger.error('Price broadcast failed: %s', exc, extra={'symbol': symbol})
                    continue
            final_now_utc = dt.datetime.now(tz=dt.timezone.utc)
            if quote is None:
                self.logger.warning('Skipping live quote: quote is unavailable', extra={'symbol': symbol})
                continue
            if not self._is_quote_fresh(quote, final_now_utc):
                age_ms = int(final_now_utc.timestamp() * 1000) - int(quote.timestamp_ms)
                self.logger.warning('Skipping stale live quote before publish (age_ms=%s)', age_ms, extra={'symbol': symbol})
                continue
            publish_ts_ms = int(quote.timestamp_ms)
            if publish_ts_ms <= 0:
                publish_ts_ms = int(final_now_utc.timestamp() * 1000)
            publish_timestamp = dt.datetime.fromtimestamp(publish_ts_ms / 1000, tz=dt.timezone.utc)
            payload = {'symbol': symbol, 'asset_class': asset_class, 'price': quote.price, 'bid': quote.bid, 'ask': quote.ask, 'timestamp': publish_timestamp.isoformat().replace('+00:00', 'Z'), 'ts': publish_ts_ms}
            try:
                await self._publish_json(exchange_name=self.exchange_name, routing_key=self.price_routing_key, payload=payload)
            except Exception as exc:
                self.logger.error('Price publish failed: %s', exc, extra={'symbol': symbol})

    async def _refresh_quotes(self, symbols: list[str]) -> None:
        unique_symbols = sorted(set(symbols))
        if not unique_symbols:
            return
        semaphore = asyncio.Semaphore(self.price_fetch_workers)

        async def _load(symbol: str) -> None:
            async with semaphore:
                try:
                    await self._fetch_quote(symbol=symbol)
                except Exception as exc:
                    self.logger.error('Quote refresh failed: %s', exc, extra={'symbol': symbol})
        await asyncio.gather(*(_load(symbol) for symbol in unique_symbols))

    async def _publish_json(self, exchange_name: str, routing_key: str, payload: dict[str, Any]) -> bool:
        validation_errors: list[str] = []
        if routing_key == self.candidate_routing_key:
            validation_errors = validate_candidate_payload(payload)
        elif routing_key == self.price_routing_key:
            validation_errors = validate_live_payload(payload, now_utc=dt.datetime.now(tz=dt.timezone.utc), max_age_ms=self.quote_freshness_ms)
        if validation_errors:
            self.logger.error('Skipping publish: invalid payload (%s)', '; '.join(validation_errors), extra={'symbol': str(payload.get('symbol', '-'))})
            return False
        body = json.dumps(payload, ensure_ascii=False)
        publish_async = getattr(self.rabbit, 'publish_async', None)
        if callable(publish_async):
            await publish_async(exchange_name=exchange_name, routing_key=routing_key, body=body)
            return True
        await self._run_blocking(self.rabbit.publish, exchange_name=exchange_name, routing_key=routing_key, body=body)
        return True

    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Any:
        return await self._run_blocking(self.collector.fetch_ohlcv, symbol, timeframe, limit)

    async def _fetch_close_series(self, symbol: str, timeframe: str, limit: int) -> Any:
        return await self._run_blocking(self.collector.fetch_close_series, symbol, timeframe, limit)

    async def _fetch_quote(self, symbol: str) -> Quote:
        quote = await self._run_blocking(self.collector.fetch_quote, symbol)
        self._quote_cache[symbol] = quote
        return quote

    async def _fetch_quote_with_cache(self, symbol: str, now_utc: dt.datetime, allow_stale: bool) -> Quote | None:
        cached_quote = self._quote_cache.get(symbol)
        if self._is_quote_fresh(cached_quote, now_utc):
            return cached_quote
        try:
            return await self._fetch_quote(symbol=symbol)
        except Exception:
            if allow_stale and cached_quote is not None:
                self.logger.warning('Using stale cached quote after fetch failure', extra={'symbol': symbol})
                return cached_quote
            raise

    def _is_quote_fresh(self, quote: Quote | None, now_utc: dt.datetime) -> bool:
        if quote is None:
            return False
        now_ms = int(now_utc.timestamp() * 1000)
        age_ms = now_ms - int(quote.timestamp_ms)
        return age_ms <= self.quote_freshness_ms

    async def _run_blocking(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        if self.use_thread_pool:
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except RuntimeError as exc:
                if not self._thread_pool_fallback_warned:
                    self.logger.warning('Thread pool unavailable, falling back to inline blocking calls: %s', exc, extra={'symbol': '-'})
                    self._thread_pool_fallback_warned = True
                self.use_thread_pool = False
        return fn(*args, **kwargs)

    def _closed_candle_view(self, indicator_df: Any, now_utc: dt.datetime, timeframe: str) -> Any:
        if len(indicator_df) < 2:
            return indicator_df
        last_idx = indicator_df.index[-1]
        if self._is_candle_closed(candle_ts=last_idx, now_utc=now_utc, timeframe=timeframe):
            return indicator_df
        return indicator_df.iloc[:-1]

    @staticmethod
    def _is_candle_closed(candle_ts: Any, now_utc: dt.datetime, timeframe: str) -> bool:
        if not isinstance(candle_ts, dt.datetime):
            return True
        if candle_ts.tzinfo is None:
            candle_ts = candle_ts.replace(tzinfo=dt.timezone.utc)
        else:
            candle_ts = candle_ts.astimezone(dt.timezone.utc)
        duration = ExternalMarketsService._timeframe_to_timedelta(timeframe)
        if duration is None:
            return True
        return candle_ts + duration <= now_utc

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> dt.timedelta | None:
        if timeframe.endswith('m') and timeframe[:-1].isdigit():
            return dt.timedelta(minutes=int(timeframe[:-1]))
        if timeframe.endswith('h') and timeframe[:-1].isdigit():
            return dt.timedelta(hours=int(timeframe[:-1]))
        if timeframe.endswith('d') and timeframe[:-1].isdigit():
            return dt.timedelta(days=int(timeframe[:-1]))
        return None

    def _build_candidate_payload(self, symbol: str, asset_class: str, features: dict[str, Any], trigger_result: TriggerResult, indicator_df: Any, patterns_payload: dict[str, Any], composite_score: float, signal_timestamp: dt.datetime) -> dict[str, Any]:
        timestamp = signal_timestamp.astimezone(dt.timezone.utc)
        return {'signal_id': str(uuid.uuid4()), 'symbol': symbol, 'asset_class': asset_class, 'timestamp': timestamp.isoformat().replace('+00:00', 'Z'), 'trigger_reason': trigger_result.reasons[0], 'trigger_reasons': list(trigger_result.reasons), 'heuristic_signal_consensus': trigger_result.heuristic_signal_consensus, 'features': features, 'indicators': trigger_result.indicators, 'patterns': patterns_payload, 'historical_snapshots': self.feature_generator.build_historical_snapshots(indicator_df, count=2), 'composite_score': round(composite_score, 4), 'entry_price_suggestion': 'market', 'signal_log_db_id': None}

    @staticmethod
    def _normalize_timestamp(value: Any) -> dt.datetime | None:
        if not isinstance(value, dt.datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    @staticmethod
    def _candidate_dedup_key(signal_timestamp: dt.datetime) -> str:
        return signal_timestamp.astimezone(dt.timezone.utc).isoformat()

    def _signal_timestamp_from_closed_candle(self, indicator_df: Any, timeframe: str) -> dt.datetime | None:
        if len(indicator_df) == 0:
            return None
        candle_open_ts = self._normalize_timestamp(indicator_df.index[-1])
        if candle_open_ts is None:
            return None
        duration = self._timeframe_to_timedelta(timeframe)
        if duration is None:
            return candle_open_ts
        return candle_open_ts + duration

    def _clamp_int_setting(self, raw_value: Any, min_value: int, max_value: int, setting_name: str) -> int:
        try:
            numeric = int(raw_value)
        except (TypeError, ValueError):
            numeric = min_value
        clamped = max(min_value, min(max_value, numeric))
        if clamped != numeric:
            self.logger.warning('Setting %s=%s out of range, clamped to %s (allowed %s..%s)', setting_name, raw_value, clamped, min_value, max_value, extra={'symbol': '-'})
        return clamped

async def _run_service() -> None:
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'settings.yml'
    settings = load_settings(config_path)
    service = ExternalMarketsService(settings)
    try:
        await service.run()
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        await service.stop()
    finally:
        await service.stop()
if __name__ == '__main__':
    try:
        asyncio.run(_run_service())
    except KeyboardInterrupt:
        pass
