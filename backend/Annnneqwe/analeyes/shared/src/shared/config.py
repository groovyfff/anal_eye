from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Загрузчик YAML-конфигурации с dot-нотацией (shared/config.py)."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = os.environ.get('ANALEYES_CONFIG', '/app/config/settings.yml')
        path = Path(config_path)
        if not path.is_file():
            fallback = Path.cwd() / 'config' / 'settings.yml'
            if fallback.is_file():
                path = fallback
            else:
                self._data: dict[str, Any] = {}
                return
        with path.open('r', encoding='utf-8') as handle:
            self._data = yaml.safe_load(handle) or {}

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self._data
        for segment in key.split('.'):
            if not isinstance(node, dict) or segment not in node:
                return default
            node = node[segment]
        return node

    def all(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def get_database_config(self) -> dict[str, Any]:
        return dict(self.get('database', {}) or {})
