from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from decimal import Decimal
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DataEncoder(json.JSONEncoder):
    """Сериализатор для RabbitMQ payload (datetime, numpy, pydantic-like объекты)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (dt.datetime, dt.date)):
            if isinstance(obj, dt.datetime) and obj.tzinfo is not None:
                return obj.astimezone(dt.timezone.utc).isoformat().replace('+00:00', 'Z')
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        return str(obj)


def dumps_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, cls=DataEncoder)
