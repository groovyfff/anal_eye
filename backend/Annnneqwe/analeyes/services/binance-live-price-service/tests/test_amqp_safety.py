from __future__ import annotations

import pytest

from src.amqp_safety import validate_rabbitmq_credentials


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
