from __future__ import annotations

import datetime as dt
import uuid

import numpy as np

from shared.utils.data_encoder import dumps_payload


def test_data_encoder_serializes_numpy_and_datetime() -> None:
    payload = {
        'signal_id': uuid.uuid4(),
        'timestamp': dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc),
        'features': {'rsi': np.float64(55.3), 'values': np.array([1.0, 2.0])},
    }
    body = dumps_payload(payload)
    assert '55.3' in body
    assert '2026-01-01' in body
    assert 'signal_id' in body
