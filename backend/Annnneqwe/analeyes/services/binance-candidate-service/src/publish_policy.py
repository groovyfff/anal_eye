from __future__ import annotations

import time


class PublishPolicy:
    """Throttle candidate publishes per symbol."""

    def __init__(
        self,
        *,
        throttle_sec: int,
        publish_on_candle_close: bool,
        publish_on_every_update: bool,
    ) -> None:
        self.throttle_sec = max(0, throttle_sec)
        self.publish_on_candle_close = publish_on_candle_close
        self.publish_on_every_update = publish_on_every_update
        self._last_publish_mono: dict[str, float] = {}

    def should_publish(self, symbol: str, *, candle_closed: bool) -> bool:
        if self.publish_on_every_update:
            return True
        if candle_closed and self.publish_on_candle_close:
            return True
        now = time.monotonic()
        last = self._last_publish_mono.get(symbol.upper())
        if last is None:
            return True
        return (now - last) >= self.throttle_sec

    def mark_published(self, symbol: str) -> None:
        self._last_publish_mono[symbol.upper()] = time.monotonic()
