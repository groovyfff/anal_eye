from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any


class _SafeContextFormatter(logging.Formatter):

    def __init__(self, fmt: str, service_name: str) -> None:
        super().__init__(fmt)
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'service_name'):
            record.service_name = self.service_name
        if not hasattr(record, 'symbol'):
            record.symbol = '-'
        return super().format(record)


class _ContextFilter(logging.Filter):

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'service_name'):
            record.service_name = self.service_name
        if not hasattr(record, 'symbol'):
            record.symbol = '-'
        return True


class _JsonFormatter(logging.Formatter):

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'service_name'):
            record.service_name = self.service_name
        if not hasattr(record, 'symbol'):
            record.symbol = '-'
        payload: dict[str, Any] = {
            'timestamp': dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
            'level': record.levelname,
            'logger': record.name,
            'service_name': str(record.service_name),
            'symbol': str(record.symbol),
            'message': record.getMessage(),
        }
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _ensure_required_context_fields(fmt: str) -> str:
    required = [('service_name', '[%(service_name)s]'), ('symbol', '[%(symbol)s]')]
    missing_prefixes = [prefix for key, prefix in required if f'%({key})s' not in fmt]
    if not missing_prefixes:
        return fmt
    return f"{''.join(missing_prefixes)} {fmt}"


def setup_logging(config: dict[str, Any]) -> None:
    level_name = str(config.get('level', 'INFO')).upper()
    service_name = config.get('service_name', 'tracker-service')
    json_logs = bool(config.get('json', False))
    configured_format = str(
        config.get('format', '[%(service_name)s][%(symbol)s] %(asctime)s %(levelname)s %(name)s: %(message)s')
    )
    log_format = _ensure_required_context_fields(configured_format)
    root = logging.getLogger()
    root.setLevel(level_name)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        formatter: logging.Formatter = _JsonFormatter(service_name=service_name)
    else:
        formatter = _SafeContextFormatter(log_format, service_name=service_name)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.addFilter(_ContextFilter(service_name=service_name))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
