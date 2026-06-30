from __future__ import annotations

import pytest

from src.amqp_safety import validate_rabbitmq_credentials
from src.publish_policy import PublishPolicy


def test_accepts_analeyes_vhost() -> None:
    user, vhost = validate_rabbitmq_credentials("amqp://analeyes:secret@rabbitmq:5672/analeyes")
    assert user == "analeyes"
    assert vhost == "analeyes"


@pytest.mark.parametrize(
    "url",
    [
        "amqp://guest:guest@rabbitmq:5672/analeyes",
        "amqp://analeyes:secret@rabbitmq:5672/",
        "amqp://analeyes:secret@rabbitmq:5672/%2F",
        "amqp://analeyes:secret@rabbitmq:5672/other",
    ],
)
def test_refuses_legacy_config(url: str) -> None:
    with pytest.raises(SystemExit, match="refused legacy RabbitMQ config"):
        validate_rabbitmq_credentials(url)


def test_publish_policy_throttle() -> None:
    policy = PublishPolicy(throttle_sec=60, publish_on_candle_close=False, publish_on_every_update=False)
    assert policy.should_publish("BTCUSDT", candle_closed=False) is True
    policy.mark_published("BTCUSDT")
    assert policy.should_publish("BTCUSDT", candle_closed=False) is False

    policy_close = PublishPolicy(throttle_sec=60, publish_on_candle_close=True, publish_on_every_update=False)
    policy_close.mark_published("BTCUSDT")
    assert policy_close.should_publish("BTCUSDT", candle_closed=True) is True
