from __future__ import annotations
from typing import Any
from src.utils.pika_client import PikaClient

class _FakeChannel:

    def __init__(self) -> None:
        self.exchange_calls: list[dict[str, Any]] = []
        self.queue_declare_calls: list[dict[str, Any]] = []
        self.queue_bind_calls: list[dict[str, Any]] = []

    def exchange_declare(self, **kwargs: Any) -> None:
        self.exchange_calls.append(kwargs)

    def queue_declare(self, **kwargs: Any) -> None:
        self.queue_declare_calls.append(kwargs)

    def queue_bind(self, **kwargs: Any) -> None:
        self.queue_bind_calls.append(kwargs)

class _FakeConnection:

    def __init__(self, channel: _FakeChannel) -> None:
        self._channel = channel
        self.is_open = True

    def channel(self) -> _FakeChannel:
        return self._channel

    def close(self) -> None:
        self.is_open = False

class _FlakyPublishChannel:

    def __init__(self, fail_first_publish: bool) -> None:
        self.fail_first_publish = fail_first_publish
        self._publish_calls = 0
        self.published: list[dict[str, Any]] = []

    def exchange_declare(self, **kwargs: Any) -> None:
        _ = kwargs

    def queue_declare(self, **kwargs: Any) -> None:
        _ = kwargs

    def queue_bind(self, **kwargs: Any) -> None:
        _ = kwargs

    def basic_publish(self, **kwargs: Any) -> None:
        self._publish_calls += 1
        if self.fail_first_publish and self._publish_calls == 1:
            raise RuntimeError('stream lost')
        self.published.append(kwargs)

def test_connect_declares_exchange_and_required_bindings(monkeypatch) -> None:
    fake_channel = _FakeChannel()

    def _fake_blocking_connection(parameters: Any) -> _FakeConnection:
        _ = parameters
        return _FakeConnection(channel=fake_channel)
    monkeypatch.setattr('src.utils.pika_client.pika.BlockingConnection', _fake_blocking_connection)
    client = PikaClient(url='amqp://guest:guest@localhost:5672/', default_exchange='analeyes_exchange', queue_bindings=[('data.candidates.ai', 'data.candidates.ai'), ('data.live_prices.external', 'data.live_prices.external')])
    client._connect_blocking()
    assert fake_channel.exchange_calls == [{'exchange': 'analeyes_exchange', 'exchange_type': 'topic', 'durable': True}]
    assert fake_channel.queue_declare_calls == [{'queue': 'data.candidates.ai', 'durable': True}, {'queue': 'data.live_prices.external', 'durable': True}]
    assert fake_channel.queue_bind_calls == [{'exchange': 'analeyes_exchange', 'queue': 'data.candidates.ai', 'routing_key': 'data.candidates.ai'}, {'exchange': 'analeyes_exchange', 'queue': 'data.live_prices.external', 'routing_key': 'data.live_prices.external'}]

def test_publish_reconnects_and_retries_once(monkeypatch) -> None:
    first_channel = _FlakyPublishChannel(fail_first_publish=True)
    second_channel = _FlakyPublishChannel(fail_first_publish=False)
    connections = [_FakeConnection(first_channel), _FakeConnection(second_channel)]

    def _fake_blocking_connection(parameters: Any) -> _FakeConnection:
        _ = parameters
        if not connections:
            raise RuntimeError('no more fake connections')
        return connections.pop(0)
    monkeypatch.setattr('src.utils.pika_client.pika.BlockingConnection', _fake_blocking_connection)
    client = PikaClient(url='amqp://guest:guest@localhost:5672/', default_exchange='analeyes_exchange', queue_bindings=[])
    client._connect_blocking()
    client._publish_blocking(exchange_name='analeyes_exchange', routing_key='data.live_prices.external', body='{}')
    assert second_channel.published
