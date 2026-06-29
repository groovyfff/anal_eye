"""RabbitMQ messaging: consume candidates, publish final signals."""

from ae_brain.messaging.rabbitmq import SignalBroker

__all__ = ["SignalBroker"]
