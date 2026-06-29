from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.IGNORECASE | re.MULTILINE)


def strip_markdown_json_fence(text: str) -> str:
    """Убирает markdown-обёртку ```json — известный баг LLM-клиентов."""
    cleaned = _JSON_FENCE_RE.sub('', text.strip())
    return cleaned.strip()


class AssetClassAwareAnalyzer:
    """Переходный анализатор: rule-based + asset-class-aware промпты (без live LLM в тестах)."""

    def __init__(self, prompts_dir: Path, min_composite: float = 0.5) -> None:
        self.prompts_dir = prompts_dir
        self.min_composite = min_composite

    def load_prompt_template(self, asset_class: str) -> str:
        asset_class = asset_class or 'crypto'
        path = self.prompts_dir / f'{asset_class}_gpt_prompt.txt'
        if not path.is_file():
            path = self.prompts_dir / 'crypto_gpt_prompt.txt'
        if path.is_file():
            return path.read_text(encoding='utf-8')
        return 'Analyze {symbol} {asset_class} with features: {features}'

    def analyze(self, candidate: dict[str, Any]) -> dict[str, Any]:
        asset_class = str(candidate.get('asset_class', 'crypto'))
        template = self.load_prompt_template(asset_class)
        features = candidate.get('features') or {}
        prompt_preview = template.replace('{symbol}', str(candidate.get('symbol', '')))
        prompt_preview = prompt_preview.replace('{asset_class}', asset_class)
        prompt_preview = prompt_preview.replace('{features}', json.dumps(features, ensure_ascii=False)[:500])

        consensus = str(candidate.get('heuristic_signal_consensus', 'NEUTRAL')).upper()
        composite = float(candidate.get('composite_score') or 0.0)
        current_price = float(features.get('current_price') or 0.0)

        if consensus not in {'LONG', 'SHORT'} or composite < self.min_composite or current_price <= 0:
            return {
                'decision': 'SKIP',
                'confidence': 0.0,
                'reason': 'Rule analyzer: weak consensus or composite below threshold',
                'prompt_preview': prompt_preview,
            }

        if consensus == 'LONG':
            tp = round(current_price * 1.02, 4)
            sl = round(current_price * 0.99, 4)
        else:
            tp = round(current_price * 0.98, 4)
            sl = round(current_price * 1.01, 4)

        return {
            'decision': consensus,
            'confidence': round(min(0.99, max(composite, 0.5)), 4),
            'reason': f'Rule analyzer aligned with heuristic {consensus}',
            'entry_price': 'market',
            'tp': tp,
            'sl': sl,
            'leverage': 1.0,
            'prompt_preview': prompt_preview,
        }
