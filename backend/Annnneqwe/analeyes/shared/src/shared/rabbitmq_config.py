"""Central RabbitMQ URL resolution for all AnalEyes services.

Preferred: set ``RABBITMQ_URL`` in ``.env`` (production / docker-compose).

Fallback: build from split variables (all required except host/port/vhost defaults)::

    RABBITMQ_USER
    RABBITMQ_PASSWORD  (legacy alias: RABBITMQ_PASS)
    RABBITMQ_HOST      (default: rabbitmq)
    RABBITMQ_PORT      (default: 5672)
    RABBITMQ_VHOST     (default: analeyes)

Password and vhost are URL-encoded for AMQP safety.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote, urlparse, urlunparse


def build_rabbitmq_url(
    *,
    user: str,
    password: str,
    host: str = "rabbitmq",
    port: str | int = 5672,
    vhost: str = "analeyes",
) -> str:
    user_q = quote(user, safe="")
    pass_q = quote(password, safe="")
    vhost_q = quote(vhost, safe="")
    return f"amqp://{user_q}:{pass_q}@{host}:{port}/{vhost_q}"


def resolve_rabbitmq_url() -> str:
    """Return the AMQP URL for application services (never use guest)."""
    explicit = (os.environ.get("RABBITMQ_URL") or "").strip()
    if explicit:
        return explicit

    user = (os.environ.get("RABBITMQ_USER") or "analeyes").strip()
    password = (os.environ.get("RABBITMQ_PASSWORD") or os.environ.get("RABBITMQ_PASS") or "").strip()
    if not password:
        raise ValueError(
            "RabbitMQ password missing: set RABBITMQ_URL or RABBITMQ_PASSWORD in the environment"
        )
    host = (os.environ.get("RABBITMQ_HOST") or "rabbitmq").strip()
    port = (os.environ.get("RABBITMQ_PORT") or "5672").strip()
    vhost = (os.environ.get("RABBITMQ_VHOST") or "analeyes").strip()
    return build_rabbitmq_url(user=user, password=password, host=host, port=port, vhost=vhost)


def sanitized_rabbitmq_url(url: str) -> str:
    """Mask password for logs."""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "<invalid-amqp-url>"
    host = parsed.hostname or ""
    port = parsed.port or 5672
    user = parsed.username or ""
    # path includes leading slash; vhost may be URL-encoded in the path
    vhost = parsed.path or "/"
    netloc = f"{user}:****@{host}:{port}" if user else f"{host}:{port}"
    return urlunparse((parsed.scheme, netloc, vhost, "", "", ""))


def rabbitmq_connection_info(url: str | None = None) -> dict[str, str]:
    """Structured connection metadata for startup / debug logs."""
    resolved = url or resolve_rabbitmq_url()
    parsed = urlparse(resolved)
    vhost_raw = parsed.path.lstrip("/")
    # urlparse leaves percent-encoding; decode for human-readable logs
    from urllib.parse import unquote

    vhost = unquote(vhost_raw) if vhost_raw else "/"
    return {
        "url_sanitized": sanitized_rabbitmq_url(resolved),
        "host": parsed.hostname or "",
        "port": str(parsed.port or 5672),
        "user": unquote(parsed.username) if parsed.username else "",
        "vhost": vhost,
    }


def inject_rabbitmq_url(settings: dict[str, Any]) -> dict[str, Any]:
    """Set ``settings['rabbitmq']['url']`` from env (mutates and returns settings)."""
    settings.setdefault("rabbitmq", {})["url"] = resolve_rabbitmq_url()
    return settings
