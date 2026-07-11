#!/usr/bin/env python3
"""Production diagnostic for the AnalEyes live candidate pipeline.

Prints a structured health report and a single verdict line:

    OK
    WEBSOCKET_NOT_RECEIVING
    RABBITMQ_ROUTING_BROKEN
    AEBRAIN_NOT_CONSUMING
    NOTIFICATION_BLOCKED

The script shells out to ``docker`` (containers + logs + rabbitmqctl) so it can
run from the host against the live compose stack. It makes no changes.

Usage::

    python scripts/diagnose_live_pipeline.py
    python scripts/diagnose_live_pipeline.py --rabbitmq-container analeyes-rabbitmq-1
    python scripts/diagnose_live_pipeline.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Container names (overridable via CLI for non-default compose project names).
DEFAULT_BINANCE_CANDIDATE = "analeyes-binance-candidate-service-1"
DEFAULT_NOTIFICATION = "analeyes-notification-service-1"
DEFAULT_AEBRAIN = "aeb-app"
DEFAULT_RABBITMQ = "analeyes-rabbitmq-1"
DEFAULT_RABBITMQ_VHOST = "analeyes"

# Log events we grep for (must match src/binance_ws.py + main.py).
BINANCE_EVENTS = [
    "websocket_url",
    "websocket_connected",
    "websocket_reconnected",
    "websocket_message_received",
    "kline_update_received",
    "candle_not_closed_skipped",
    "candle_closed_received",
    "candidate_publish_allowed",
    "candidate_dedup_skipped",
    "candidate_publish_failed",
    "websocket_idle_reconnect",
    "websocket_error",
    "rabbitmq_publish_config",
    "rest_backfill_loaded",
    "rest_fallback_started",
    "rest_fallback_enabled",
    "rest_fallback_poll",
    "rest_fallback_latest_closed_candle",
    "rest_fallback_candidate_publish_allowed",
    "rest_fallback_candidate_dedup_skipped",
    "rest_fallback_no_new_closed_candle",
    "rest_fallback_error",
    "ws_marked_unhealthy",
    "ws_marked_healthy",
]
REST_EVENTS = [
    "rest_fallback_started",
    "rest_fallback_poll",
    "rest_fallback_latest_closed_candle",
    "rest_fallback_candidate_publish_allowed",
    "rest_fallback_candidate_dedup_skipped",
    "rest_fallback_no_new_closed_candle",
    "rest_fallback_error",
    "ws_marked_unhealthy",
    "ws_marked_healthy",
]
AEBRAIN_EVENTS = [
    "AEBrain received candidate",
    "AEBrain consumer registered",
    "AEBrain published signal.final",
    "AEBrain suppressed signal",
    "AEBrain SKIP candidate",
]

# Threshold: if no kline frame seen within this many seconds, WS is stale.
WS_STALE_AFTER_SEC = 90 * 60  # 1.5h (1h candles only close once per hour)


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)


def _run(cmd: list[str], *, timeout: float = 15.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"<timeout after {timeout}s>"


def _docker_ps() -> dict[str, bool]:
    rc, out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"])
    running: dict[str, bool] = {}
    if rc != 0:
        return running
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            running[parts[0]] = "Up" in parts[1]
    return running


def _container_logs(name: str, *, since: str = "6h") -> str:
    if not shutil.which("docker"):
        return ""
    rc, out = _run(["docker", "logs", "--since", since, name], timeout=20.0)
    return out if rc == 0 else ""


def _last_event_ts(logs: str, needle: str) -> str | None:
    """Return the timestamp prefix of the last line containing needle, or None."""
    last: str | None = None
    for line in logs.splitlines():
        if needle in line:
            last = line
    if last is None:
        return None
    # Logs look like: "2026-07-05 12:00:00,123 INFO [src.binance_ws] websocket_url ..."
    m = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+)", last)
    return m.group(1).replace(",", ".") if m else None


def _parse_log_ts(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _container_env(name: str, keys: list[str]) -> dict[str, str]:
    rc, out = _run(["docker", "exec", name, "env"], timeout=10.0)
    env: dict[str, str] = {}
    if rc != 0:
        return env
    wanted = {k.upper() for k in keys}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.upper() in wanted:
            env[k] = v
    return env


def _queue_stats(rabbitmq_container: str, vhost: str) -> dict[str, dict[str, str]]:
    rc, out = _run(
        ["docker", "exec", rabbitmq_container, "rabbitmqctl", "-p", vhost, "list_queues",
         "name", "messages", "consumers"],
        timeout=20.0,
    )
    stats: dict[str, dict[str, str]] = {}
    if rc != 0:
        return stats
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("q_"):
            stats[parts[0]] = {"messages": parts[1], "consumers": parts[2]}
    return stats


def check_containers(names: dict[str, str]) -> tuple[Section, dict[str, bool]]:
    sec = Section("CONTAINERS")
    running = _docker_ps()
    state: dict[str, bool] = {}
    for role, name in names.items():
        up = running.get(name, False)
        state[role] = up
        sec.lines.append(f"  {role:24s} {name:42s} {'RUNNING' if up else 'NOT RUNNING'}")
    return sec, state


def check_rabbitmq(container: str, vhost: str) -> tuple[Section, dict[str, dict[str, str]]]:
    sec = Section(f"RABBITMQ (container={container} vhost={vhost})")
    stats = _queue_stats(container, vhost)
    if not stats:
        sec.lines.append("  <could not query rabbitmqctl — is the container running?>")
        return sec, stats
    for q in ("q_data_candidates_ai", "q_new_signals"):
        s = stats.get(q)
        if s:
            sec.lines.append(f"  {q:28s} messages={s['messages']:>6s} consumers={s['consumers']:>3s}")
        else:
            sec.lines.append(f"  {q:28s} <missing — not declared>")
    return sec, stats


def check_binance_candidate(name: str) -> tuple[Section, dict[str, str | None], dict[str, str]]:
    sec = Section(f"BINANCE-CANDIDATE-SERVICE (container={name})")
    env = _container_env(name, [
        "SYMBOLS", "CANDIDATE_TIMEFRAME", "CANDIDATE_CLOSED_CANDLES_ONLY",
        "CANDIDATE_DEDUP_ENABLED", "CANDIDATE_WINDOW_CANDLES",
        "ENABLE_LEGACY_PARSER", "ENABLE_HIGH_FREQUENCY_TEST_PARSER",
        "CANDIDATE_CONTINUOUS_TEST_MODE",
        "BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE",
        "BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE",
        "CANDIDATE_REST_FALLBACK_ENABLED",
        "CANDIDATE_REST_FALLBACK_POLL_SEC",
        "CANDIDATE_REST_FALLBACK_ALWAYS_ON",
        "CANDIDATE_WS_IDLE_TIMEOUT_SEC",
    ])
    for k in sorted(env):
        sec.lines.append(f"  {k}={env[k]}")
    logs = _container_logs(name)
    last_events = {ev: _last_event_ts(logs, ev) for ev in BINANCE_EVENTS}
    sec.lines.append("  -- last log events --")
    for ev in BINANCE_EVENTS:
        ts = last_events[ev]
        sec.lines.append(f"  {ev:38s} {ts or '<never>'}")
    return sec, last_events, env


def check_aebrain(name: str) -> tuple[Section, dict[str, str | None]]:
    sec = Section(f"AE-BRAIN (container={name})")
    env = _container_env(name, [
        "AEB_ALLOWED_SYMBOLS", "AEB_MIN_PUBLISH_CONFIDENCE", "AEB_ONLY_BTC",
        "AEB_PUBLISH_SKIPPED_DECISIONS", "AEB_INPUT_QUEUE", "AEB_INPUT_ROUTING_KEY",
        "AEB_INPUT_EXCHANGE", "AEB_OUTPUT_ROUTING_KEY",
    ])
    for k in sorted(env):
        sec.lines.append(f"  {k}={env[k]}")
    logs = _container_logs(name)
    last_events = {ev: _last_event_ts(logs, ev) for ev in AEBRAIN_EVENTS}
    sec.lines.append("  -- last log events --")
    for ev in AEBRAIN_EVENTS:
        ts = last_events[ev]
        sec.lines.append(f"  {ev:36s} {ts or '<never>'}")
    return sec, last_events


def check_notification(name: str) -> Section:
    sec = Section(f"NOTIFICATION-SERVICE (container={name})")
    env = _container_env(name, [
        "NOTIFICATION_MIN_CONFIDENCE", "NOTIFICATION_SEND_SKIPPED_DECISIONS",
        "ANAL_EYES_ALLOWED_SYMBOLS",
    ])
    for k in sorted(env):
        sec.lines.append(f"  {k}={env[k]}")
    return sec


def _verdict(
    containers: dict[str, bool],
    queues: dict[str, dict[str, str]],
    binance_events: dict[str, str | None],
    aebrain_events: dict[str, str | None],
    rest_enabled_env: bool | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    binance_up = containers.get("binance-candidate", False)
    aebrain_up = containers.get("ae-brain", False)
    rabbit_up = containers.get("rabbitmq", False)
    notif_up = containers.get("notification", False)

    # --- WebSocket health -----------------------------------------------------
    msg_ts = binance_events.get("websocket_message_received")
    kline_ts = binance_events.get("kline_update_received")
    ws_ever_received = bool(msg_ts or kline_ts)
    ws_stale = False
    if ws_ever_received and kline_ts:
        dt = _parse_log_ts(kline_ts)
        if dt and (datetime.now(timezone.utc) - dt).total_seconds() > WS_STALE_AFTER_SEC:
            ws_stale = True
    ws_healthy = ws_ever_received and not ws_stale

    # --- REST fallback health -------------------------------------------------
    rest_started = bool(binance_events.get("rest_fallback_started"))
    rest_poll_ts = binance_events.get("rest_fallback_poll")
    rest_recently_polled = False
    if rest_poll_ts:
        dt = _parse_log_ts(rest_poll_ts)
        # Consider REST alive if it polled within the last 5 minutes.
        if dt and (datetime.now(timezone.utc) - dt).total_seconds() <= 5 * 60:
            rest_recently_polled = True
    rest_published = bool(binance_events.get("rest_fallback_candidate_publish_allowed"))
    rest_dedup = bool(binance_events.get("rest_fallback_candidate_dedup_skipped"))
    rest_alive = rest_started and rest_recently_polled

    # --- RabbitMQ routing -----------------------------------------------------
    candidate_q = queues.get("q_data_candidates_ai")
    rabbitmq_broken = False
    if rabbit_up and candidate_q is None:
        rabbitmq_broken = True
        reasons.append("RABBITMQ_ROUTING_BROKEN: q_data_candidates_ai not declared")

    # --- AE Brain consuming ---------------------------------------------------
    aebrain_consuming = bool(aebrain_events.get("AEBrain consumer registered"))
    aebrain_received = bool(aebrain_events.get("AEBrain received candidate"))

    # --- Verdict priority -----------------------------------------------------
    if rabbitmq_broken:
        return "RABBITMQ_ROUTING_BROKEN", reasons

    if binance_up and not ws_healthy:
        # WebSocket is dead. Does REST fallback cover for it?
        if rest_enabled_env is False and not rest_started:
            return (
                "WEBSOCKET_NOT_RECEIVING_AND_NO_REST_FALLBACK",
                [
                    "WEBSOCKET_NOT_RECEIVING_AND_NO_REST_FALLBACK",
                    "detail: WS not receiving and CANDIDATE_REST_FALLBACK_ENABLED=false",
                    "detail: no websocket_message_received / kline_update_received, no REST fallback",
                ],
            )
        if not rest_started:
            return (
                "WEBSOCKET_NOT_RECEIVING_AND_NO_REST_FALLBACK",
                [
                    "WEBSOCKET_NOT_RECEIVING_AND_NO_REST_FALLBACK",
                    "detail: WS not receiving and no rest_fallback_started log",
                ],
            )
        if not rest_recently_polled:
            return (
                "REST_FALLBACK_NOT_POLLING",
                [
                    "REST_FALLBACK_NOT_POLLING",
                    "detail: rest_fallback_started observed but no recent rest_fallback_poll",
                    f"detail: last rest_fallback_poll={rest_poll_ts or '<never>'}",
                ],
            )
        # REST is polling. If AE Brain is not consuming despite REST publishing, flag it.
        if aebrain_up and not aebrain_consuming:
            return (
                "AEBRAIN_NOT_CONSUMING",
                [
                    "AEBRAIN_NOT_CONSUMING",
                    "detail: REST fallback alive but AE Brain has no consumer registered",
                ],
            )
        if rest_published or rest_dedup:
            return (
                "OK_REST_FALLBACK",
                [
                    "OK_REST_FALLBACK",
                    "detail: WebSocket not receiving but REST fallback is publishing/dedup-ing closed candles",
                ],
            )
        return (
            "OK_REST_FALLBACK",
            [
                "OK_REST_FALLBACK",
                "detail: WebSocket not receiving; REST fallback is polling (no new closed candle yet)",
            ],
        )

    # WebSocket healthy.
    if ws_healthy and rest_alive:
        return "OK_WS", ["OK_WS", "detail: WebSocket receiving and REST fallback armed"]
    if ws_healthy:
        return "OK_WS", ["OK_WS", "detail: WebSocket receiving frames"]

    if binance_up and rest_alive and (rest_published or rest_dedup):
        return "OK_REST_FALLBACK", ["OK_REST_FALLBACK", "detail: REST fallback delivering closed candles"]

    # AE Brain consumer missing even when sources are fine.
    if aebrain_up and not aebrain_consuming and binance_up:
        return (
            "AEBRAIN_NOT_CONSUMING",
            ["AEBRAIN_NOT_CONSUMING", "detail: no 'AEBrain consumer registered' log"],
        )

    return "OK", ["all pipeline stages reporting healthy"]


def render(sections: list[Section], verdict: str, reasons: list[str], *, as_json: bool) -> str:
    if as_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": [asdict(s) for s in sections],
            "verdict": verdict,
            "reasons": reasons,
        }
        return json.dumps(payload, indent=2)
    out: list[str] = []
    out.append(f"=== AnalEyes live pipeline diagnostic @ {datetime.now(timezone.utc).isoformat()} ===")
    for s in sections:
        out.append(f"\n--- {s.title} ---")
        out.extend(s.lines)
    out.append("\n" + "=" * 64)
    out.append(f"VERDICT: {verdict}")
    for r in reasons:
        out.append(f"  - {r}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binance-candidate", default=DEFAULT_BINANCE_CANDIDATE)
    ap.add_argument("--notification", default=DEFAULT_NOTIFICATION)
    ap.add_argument("--aebrain", default=DEFAULT_AEBRAIN)
    ap.add_argument("--rabbitmq-container", default=DEFAULT_RABBITMQ)
    ap.add_argument("--rabbitmq-vhost", default=DEFAULT_RABBITMQ_VHOST)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("docker not found on PATH — install docker/colima and retry.", file=sys.stderr)
        return 2

    names = {
        "rabbitmq": args.rabbitmq_container,
        "binance-candidate": args.binance_candidate,
        "ae-brain": args.aebrain,
        "notification": args.notification,
    }

    sec_c, container_state = check_containers(names)
    sec_r, queue_stats = check_rabbitmq(args.rabbitmq_container, args.rabbitmq_vhost)
    sec_b, binance_events, binance_env = check_binance_candidate(args.binance_candidate)
    sec_a, aebrain_events = check_aebrain(args.aebrain)
    sec_n = check_notification(args.notification)

    rest_enabled_raw = (binance_env.get("CANDIDATE_REST_FALLBACK_ENABLED") or "").strip().lower()
    rest_enabled_env: bool | None = None
    if rest_enabled_raw:
        rest_enabled_env = rest_enabled_raw in {"1", "true", "yes", "on"}

    verdict, reasons = _verdict(
        container_state, queue_stats, binance_events, aebrain_events, rest_enabled_env=rest_enabled_env
    )
    text = render([sec_c, sec_r, sec_b, sec_a, sec_n], verdict, reasons, as_json=args.json)
    print(text)
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
