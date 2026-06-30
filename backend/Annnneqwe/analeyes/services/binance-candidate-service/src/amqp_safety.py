from __future__ import annotations

from urllib.parse import unquote, urlparse


def validate_rabbitmq_credentials(url: str) -> tuple[str, str]:
    """Refuse guest or non-analeyes vhost before connecting."""
    parsed = urlparse(url)
    user = unquote(parsed.username or "")
    vhost_raw = (parsed.path or "").lstrip("/")
    vhost = unquote(vhost_raw) if vhost_raw else "/"

    invalid_vhost = vhost in {"", "/", "%2F"} or vhost != "analeyes"
    if user == "guest" or invalid_vhost:
        display_vhost = vhost if vhost not in {"", "%2F"} else "/"
        raise SystemExit(
            "Binance candidate publisher refused legacy RabbitMQ config: "
            f"expected user=analeyes vhost=analeyes, got user={user or '<missing>'} vhost={display_vhost}"
        )
    return user, vhost
