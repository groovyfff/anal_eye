from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv

from shared.rabbitmq_config import inject_rabbitmq_url

def load_settings(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    settings_path = Path(path)
    payload = yaml.safe_load(settings_path.read_text(encoding='utf-8')) or {}
    try:
        inject_rabbitmq_url(payload)
    except ValueError:
        pass
    api_key = os.getenv('DATA_PROVIDER_API_KEY')
    if api_key:
        payload.setdefault('data_provider', {})['api_key'] = api_key
    return payload

def get_watchlist_items(settings: dict[str, Any]) -> list[dict[str, Any]]:
    watchlist = settings.get('watchlist', {})
    items: list[dict[str, Any]] = []
    mapping = {'stocks': 'stock', 'metals': 'metal', 'forex': 'forex', 'indices': 'stock'}
    for bucket, asset_class in mapping.items():
        for item in watchlist.get(bucket, []):
            normalized = {'symbol': item['symbol'], 'name': item.get('name', item['symbol']), 'asset_class': asset_class, 'use_for_correlation': bool(item.get('use_for_correlation', False)), 'instrument_type': 'index' if bucket == 'indices' else asset_class}
            items.append(normalized)
    return items
