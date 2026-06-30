"""AMQP URL parsing, masking, and legacy vhost safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse

EXPECTED_AMQP_USER = "analeyes"
EXPECTED_AMQP_VHOST = "analeyes"


@dataclass(frozen=True, slots=True)
class AmqpEndpoint:
    user: str
    password: str
    host: str
    port: int
    vhost: str
    url: str

    @property
    def masked_url(self) -> str:
        safe_pass = "***" if self.password else ""
        vhost_path = quote(self.vhost, safe="")
        return f"amqp://{self.user}:{safe_pass}@{self.host}:{self.port}/{vhost_path}"


def parse_amqp_url(url: str) -> AmqpEndpoint:
    parsed = urlparse(url)
    if parsed.scheme not in ("amqp", "amqps"):
        raise ValueError(f"unsupported AMQP scheme: {parsed.scheme!r}")
    raw_vhost = parsed.path or "/"
    if raw_vhost.startswith("/"):
        vhost = unquote(raw_vhost[1:]) or "/"
    else:
        vhost = unquote(raw_vhost) or "/"
    return AmqpEndpoint(
        user=parsed.username or "",
        password=parsed.password or "",
        host=parsed.hostname or "localhost",
        port=parsed.port or 5672,
        vhost=vhost,
        url=url,
    )


def is_invalid_vhost(vhost: str) -> bool:
    if not vhost or vhost in {"/", "%2F"}:
        return True
    return False


def assert_analeyes_amqp(endpoint: AmqpEndpoint, *, allow_legacy: bool, label: str) -> None:
    """Refuse guest/`/` and require user=vhost=analeyes unless legacy override is set."""
    if allow_legacy:
        return
    user = endpoint.user or ""
    vhost = endpoint.vhost
    if user == EXPECTED_AMQP_USER and vhost == EXPECTED_AMQP_VHOST:
        return
    raise RuntimeError(
        "AEBrain refused legacy RabbitMQ config: "
        f"expected user={EXPECTED_AMQP_USER} vhost={EXPECTED_AMQP_VHOST}, "
        f"got user={user} vhost={vhost} ({label} url={endpoint.masked_url}). "
        "Set AEB_ALLOW_LEGACY_GUEST_VHOST=true only for local debugging."
    )


# Backward-compatible alias used by existing imports.
def assert_not_legacy(endpoint: AmqpEndpoint, *, allow_legacy: bool, label: str) -> None:
    assert_analeyes_amqp(endpoint, allow_legacy=allow_legacy, label=label)


def log_endpoint(prefix: str, endpoint: AmqpEndpoint, *, exchange: str, queue: str = "", routing_key: str) -> str:
    return (
        f"AEBrain AMQP {prefix} user={endpoint.user} host={endpoint.host} port={endpoint.port} "
        f"vhost={endpoint.vhost} exchange={exchange}"
        + (f" queue={queue}" if queue else "")
        + f" routing_key={routing_key}"
    )
