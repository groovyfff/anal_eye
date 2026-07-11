"""RabbitMQ broker integration using aio-pika.

Contract
--------
* **Consume** trade candidates from ``analeyes.events`` / ``data.candidates.ai``.
* **Publish** finalized signals to ``analeyes.events`` / ``signal.final``.

Input and output may use separate brokers (e.g. test_task vs main AnalEyes).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import orjson

from ae_brain.config import AmqpInputConfig, AmqpOutputConfig, TelegramDebugConfig
from ae_brain.contracts import Decision, FinalSignal, TradeCandidate
from ae_brain.messaging.amqp_utils import assert_analeyes_amqp, log_endpoint, parse_amqp_url
from ae_brain.messaging.candidate_normalizer import normalize_candidate
from ae_brain.messaging.publish_gate import (
    default_allowed_symbol_set,
    evaluate_publish,
    is_symbol_allowed,
    normalize_candidate_symbol,
    normalize_confidence,
)
from ae_brain.messaging.skip_reason import extract_skip_reason
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.amqp")

SignalHandler = Callable[[TradeCandidate], Awaitable[FinalSignal | None]]


def _build_reason_summary(signal: FinalSignal) -> str:
    components = signal.components or {}
    source = components.get("decision_source", "ae_brain")
    ev_usd = signal.expected_value_usd
    return (
        f"AE Brain {source}: {signal.decision.value} on {signal.symbol} "
        f"(asset={signal.asset}, confidence={signal.confidence:.3f}, ev_usd={ev_usd:.2f})"
    )


def build_signal_final_payload(signal: FinalSignal, candidate: TradeCandidate) -> dict:
    payload = signal.to_dict()
    normalized_confidence = normalize_confidence(signal.confidence)
    if normalized_confidence is not None:
        payload["confidence"] = normalized_confidence
    payload["source_ai"] = "ae_brain"
    payload["signal_type"] = signal.decision.value
    payload["reason_summary"] = _build_reason_summary(signal)
    payload["tp_price"] = signal.take_profit
    payload["sl_price"] = signal.stop_loss
    payload["consensus_achieved"] = signal.decision in (Decision.LONG, Decision.SHORT)
    if signal.decision == Decision.SKIP:
        payload["skip_reason"] = extract_skip_reason(signal)
    payload["features"] = candidate.meta.get("features") or {}
    payload["candles"] = candidate.candles
    if candidate.meta.get("composite_score") is not None:
        payload["composite_score"] = candidate.meta["composite_score"]
    timing = (signal.components or {}).get("execution_timing") or {}
    for key in (
        "signal_candle_open_time",
        "signal_candle_close_time",
        "execution_time",
        "execution_price_source",
        "execution_price",
        "signal_reference_price",
    ):
        value = getattr(signal, key, None) or timing.get(key)
        if value not in (None, "", 0.0):
            payload[key] = value
    return payload


class SignalBroker:
    def __init__(
        self,
        input_cfg: AmqpInputConfig,
        output_cfg: AmqpOutputConfig,
        *,
        allow_legacy_guest_vhost: bool = False,
        min_composite_score: float = 0.0,
        models_loaded: Callable[[], bool] | None = None,
        telegram_cfg: TelegramDebugConfig | None = None,
        publish_skipped_decisions: bool = False,
        disable_signal_dedup_in_test_mode: bool = False,
        allowed_symbols: frozenset[str] | None = None,
        min_publish_confidence: float = 0.70,
        only_btc: bool = False,
    ) -> None:
        self._input_cfg = input_cfg
        self._output_cfg = output_cfg
        self._allow_legacy = allow_legacy_guest_vhost
        self._min_composite_score = min_composite_score
        self._models_loaded = models_loaded or (lambda: True)
        self._telegram_cfg = telegram_cfg or TelegramDebugConfig()
        self._publish_skipped_decisions = publish_skipped_decisions
        self._disable_signal_dedup = disable_signal_dedup_in_test_mode
        self._allowed_symbols = allowed_symbols or default_allowed_symbol_set()
        self._min_publish_confidence = min_publish_confidence
        self._only_btc = only_btc
        self._input_connection = None
        self._output_connection = None
        self._input_channel = None
        self._output_channel = None
        self._output_exchange = None
        self._input_endpoint = parse_amqp_url(input_cfg.resolved_url)
        self._output_endpoint = parse_amqp_url(output_cfg.resolved_url)

    async def connect(self) -> None:
        import aio_pika

        assert_analeyes_amqp(self._input_endpoint, allow_legacy=self._allow_legacy, label="input")
        assert_analeyes_amqp(self._output_endpoint, allow_legacy=self._allow_legacy, label="output")

        self._input_connection = await aio_pika.connect_robust(self._input_endpoint.url)
        self._input_channel = await self._input_connection.channel()
        await self._input_channel.set_qos(prefetch_count=self._input_cfg.prefetch_count)

        input_exchange = await self._input_channel.declare_exchange(
            self._input_cfg.exchange,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )
        queue = await self._input_channel.declare_queue(self._input_cfg.queue, durable=True)
        await queue.bind(input_exchange, routing_key=self._input_cfg.routing_key)

        self._output_connection = await aio_pika.connect_robust(self._output_endpoint.url)
        self._output_channel = await self._output_connection.channel()
        self._output_exchange = await self._output_channel.declare_exchange(
            self._output_cfg.exchange,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

        log.info(log_endpoint("input", self._input_endpoint, exchange=self._input_cfg.exchange, queue=self._input_cfg.queue, routing_key=self._input_cfg.routing_key))
        log.info(log_endpoint("output", self._output_endpoint, exchange=self._output_cfg.exchange, routing_key=self._output_cfg.routing_key))
        log.info(
            "AEBrain output publisher",
            exchange=self._output_cfg.exchange,
            routing_key=self._output_cfg.routing_key,
            output_host=self._output_endpoint.host,
            output_vhost=self._output_endpoint.vhost,
            input_amqp_url_masked=self._input_endpoint.masked_url,
            consumer_registered=True,
        )

    async def close(self) -> None:
        for conn in (self._input_connection, self._output_connection):
            if conn is not None:
                await conn.close()
        log.info("amqp.closed")

    async def publish_signal(self, signal: FinalSignal, candidate: TradeCandidate) -> None:
        import aio_pika

        body = orjson.dumps(build_signal_final_payload(signal, candidate))
        message = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=signal.correlation_id or None,
        )
        await self._output_exchange.publish(message, routing_key=self._output_cfg.routing_key)
        log.info(
            "AEBrain published signal.final",
            symbol=signal.symbol,
            decision=signal.decision.value,
            confidence=normalize_confidence(signal.confidence),
            source_ai="ae_brain",
            exchange=self._output_cfg.exchange,
            routing_key=self._output_cfg.routing_key,
            output_host=self._output_endpoint.host,
            output_vhost=self._output_endpoint.vhost,
            event="signal_final_published",
        )

    def _allowed_symbols_label(self) -> str:
        return ",".join(sorted(self._allowed_symbols))

    def _should_publish_signal(self, signal: FinalSignal, candidate: TradeCandidate) -> tuple[bool, str | None, float | None]:
        if self._disable_signal_dedup:
            log.info(
                "signal_dedup_bypassed",
                symbol=signal.symbol,
                decision=signal.decision.value,
                reason="AEB_DISABLE_SIGNAL_DEDUP_IN_TEST_MODE",
            )
        should_publish, reason, normalized_confidence = evaluate_publish(
            signal,
            allowed_symbols=self._allowed_symbols,
            min_confidence=self._min_publish_confidence,
        )
        if should_publish:
            return True, None, normalized_confidence
        if reason == "skip_decision" and self._publish_skipped_decisions:
            return True, None, normalized_confidence
        return False, reason, normalized_confidence

    async def consume(self, handler: SignalHandler) -> None:
        import aio_pika

        queue = await self._input_channel.declare_queue(self._input_cfg.queue, durable=True)

        async def _on_message(message: "aio_pika.abc.AbstractIncomingMessage") -> None:
            acked = False
            symbol = ""
            normalized_summary = ""
            try:
                body_size = len(message.body)
                valid_json = True
                payload: object
                try:
                    payload = orjson.loads(message.body)
                except orjson.JSONDecodeError:
                    valid_json = False
                    payload = {}
                    log.info(
                        "AEBrain received candidate",
                        delivery_tag=message.delivery_tag,
                        exchange=message.exchange,
                        routing_key=message.routing_key,
                        body_size=body_size,
                        keys=[],
                        valid_json=False,
                    )
                    log.info(
                        "AEBrain SKIP candidate",
                        symbol="",
                        reason="invalid_json",
                        normalized="{}",
                        ack="skipped_and_acked",
                    )
                    await message.ack()
                    acked = True
                    return

                top_keys = list(payload.keys()) if isinstance(payload, dict) else []
                raw_symbol = str(payload.get("symbol", "")).strip() if isinstance(payload, dict) else ""
                log.info(
                    "AEBrain received candidate",
                    symbol=raw_symbol,
                    delivery_tag=message.delivery_tag,
                    exchange=message.exchange,
                    routing_key=message.routing_key,
                    body_size=body_size,
                    keys=top_keys,
                    valid_json=valid_json,
                )

                norm = normalize_candidate(payload, min_composite_score=self._min_composite_score)
                symbol = norm.symbol
                normalized_summary = norm.summary
                if norm.skip_reason:
                    log.info(
                        "AEBrain skipped candidate",
                        symbol=symbol or "",
                        reason=norm.skip_reason,
                        normalized=normalized_summary,
                        ack="skipped_and_acked",
                    )
                    await message.ack()
                    acked = True
                    return

                if norm.payload is not None:
                    normalized_symbol = normalize_candidate_symbol(norm.payload.get("symbol"), only_btc=self._only_btc)
                    if normalized_symbol:
                        norm.payload["symbol"] = normalized_symbol
                        symbol = normalized_symbol

                if not symbol or not is_symbol_allowed(symbol, self._allowed_symbols):
                    allowed = self._allowed_symbols_label()
                    log.info(
                        "candidate_rejected_symbol",
                        symbol=symbol or raw_symbol or "",
                        reason="unsupported_symbol",
                        allowed=allowed,
                        ack="skipped_and_acked",
                    )
                    await message.ack()
                    acked = True
                    return

                if not self._models_loaded():
                    log.info(
                        "AEBrain SKIP candidate",
                        symbol=symbol,
                        reason="model_not_loaded",
                        normalized=normalized_summary,
                        ack="skipped_and_acked",
                    )
                    await message.ack()
                    acked = True
                    return

                candidate = TradeCandidate.from_message(norm.payload or {})
                features = candidate.meta.get("features") or {}
                log.info(
                    "AEBrain normalized candidate",
                    symbol=candidate.symbol,
                    candles_count=len(candidate.candles),
                    features_count=len(features),
                )

                log.info("AEBrain running analysis", symbol=candidate.symbol)
                try:
                    signal = await handler(candidate)
                except Exception as exc:
                    log.exception(
                        "AEBrain SKIP candidate",
                        symbol=candidate.symbol,
                        reason="analyzer_exception",
                        normalized=normalized_summary,
                        err=str(exc),
                    )
                    await message.nack(requeue=self._input_cfg.requeue_on_error)
                    acked = True
                    log.info("nacked_requeue", symbol=candidate.symbol)
                    return

                if signal is None:
                    log.info(
                        "AEBrain decided SKIP",
                        symbol=candidate.symbol,
                        reason="analyzer_returned_none",
                    )
                    await message.ack()
                    acked = True
                    log.info("skipped_and_acked", symbol=candidate.symbol)
                    return

                log.info(
                    "AEBrain result",
                    symbol=signal.symbol,
                    decision=signal.decision.value,
                    confidence=normalize_confidence(signal.confidence),
                )

                should_publish, suppress_reason, normalized_confidence = self._should_publish_signal(signal, candidate)
                if not should_publish:
                    if suppress_reason == "unsupported_symbol":
                        log.info(
                            "AEBrain skipped candidate",
                            symbol=signal.symbol,
                            reason="unsupported_symbol",
                            allowed=self._allowed_symbols_label(),
                            ack="skipped_and_acked",
                        )
                    elif suppress_reason == "confidence_below_threshold":
                        log.info(
                            "AEBrain suppressed signal",
                            symbol=signal.symbol,
                            decision=signal.decision.value,
                            confidence=normalized_confidence,
                            min_confidence=self._min_publish_confidence,
                            reason="confidence_below_threshold",
                        )
                    elif suppress_reason in {"skip_decision", "empty_decision"}:
                        log.info(
                            "AEBrain suppressed signal",
                            symbol=signal.symbol,
                            decision=signal.decision.value,
                            confidence=normalized_confidence,
                            reason=suppress_reason,
                        )
                    elif suppress_reason in {"negative_ev", "invalid_sizing"}:
                        log.info(
                            "AEBrain suppressed signal",
                            symbol=signal.symbol,
                            decision=signal.decision.value,
                            confidence=normalized_confidence,
                            expected_value_usd=signal.expected_value_usd,
                            reason=suppress_reason,
                        )
                    else:
                        log.info(
                            "AEBrain suppressed signal",
                            symbol=signal.symbol,
                            decision=signal.decision.value,
                            confidence=normalized_confidence,
                            reason=suppress_reason or "not_publishable",
                        )
                    await message.ack()
                    acked = True
                    log.info("skipped_and_acked", symbol=signal.symbol)
                    return

                if normalized_confidence is not None:
                    signal.confidence = normalized_confidence

                if signal.decision == Decision.SKIP and self._publish_skipped_decisions:
                    log.info(
                        "skipped_decision_published",
                        enabled=True,
                        symbol=signal.symbol,
                        skip_reason=extract_skip_reason(signal),
                    )

                try:
                    await self.publish_signal(signal, candidate)
                except Exception as exc:
                    log.error(
                        "AEBrain SKIP candidate",
                        symbol=signal.symbol,
                        reason="output_publish_failed",
                        normalized=normalized_summary,
                        err=str(exc),
                    )
                    await message.nack(requeue=True)
                    acked = True
                    log.info("nacked_requeue", symbol=signal.symbol, reason="output_publish_failed")
                    return

                if self._telegram_cfg.enabled:
                    from ae_brain.messaging.telegram_debug import send_debug_telegram

                    should_debug, debug_reason, _ = evaluate_publish(
                        signal,
                        allowed_symbols=self._allowed_symbols,
                        min_confidence=self._min_publish_confidence,
                    )
                    if should_debug:
                        await send_debug_telegram(self._telegram_cfg, build_signal_final_payload(signal, candidate))
                    else:
                        log.info(
                            "telegram_debug_suppressed",
                            symbol=signal.symbol,
                            reason=debug_reason,
                        )

                await message.ack()
                acked = True
                log.info("acked", symbol=signal.symbol)
            except Exception as exc:
                log.exception("amqp.handler_error", symbol=symbol, err=str(exc))
                if not acked:
                    await message.nack(requeue=self._input_cfg.requeue_on_error)
                    acked = True
                    log.info("nacked_requeue", symbol=symbol or "")
            finally:
                if not acked:
                    try:
                        await message.nack(requeue=self._input_cfg.requeue_on_error)
                        log.info("nacked_requeue", symbol=symbol or "", reason="finally_guard")
                    except Exception:
                        log.error("amqp.ack_failed", delivery_tag=message.delivery_tag)

        await queue.consume(_on_message, consumer_tag=self._input_cfg.consumer_tag)
        log.info(
            "AEBrain consumer registered",
            queue=self._input_cfg.queue,
            routing_key=self._input_cfg.routing_key,
            exchange=self._input_cfg.exchange,
            consumer_tag=self._input_cfg.consumer_tag,
        )
        await asyncio.Future()
