from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.formatting.chart_renderer import ChartRenderer
from src.formatting.deterministic_formatter import DeterministicSignalFormatter, extract_signal_facts
from src.formatting.llm_formatter import OpenRouterCommentFormatter
from src.formatting.presentation_config import PresentationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PresentedSignal:
    caption: str
    chart_bytes: bytes | None


class SignalPresenter:
    """Orchestrate chart rendering, optional LLM comment, and deterministic caption assembly."""

    def __init__(
        self,
        config: PresentationConfig | None = None,
        *,
        deterministic: DeterministicSignalFormatter | None = None,
        llm_formatter: OpenRouterCommentFormatter | None = None,
        chart_renderer: ChartRenderer | None = None,
    ) -> None:
        self._config = config or PresentationConfig.from_env()
        self._deterministic = deterministic or DeterministicSignalFormatter()
        self._llm = llm_formatter or OpenRouterCommentFormatter(self._config, deterministic=self._deterministic)
        self._chart = chart_renderer or ChartRenderer()

    async def present(self, payload: dict[str, Any]) -> PresentedSignal:
        symbol = payload.get("symbol")
        logger.info("signal_presentation_started symbol=%s", symbol)

        facts = extract_signal_facts(payload)
        chart_bytes = self._chart.render(payload)
        comment = await self._llm.generate_comment(facts)
        caption = self._deterministic.build_caption(facts, comment)
        caption = self._deterministic.truncate_caption_for_telegram(caption)

        logger.info(
            "signal_caption_built symbol=%s chars=%s has_chart=%s",
            symbol,
            len(caption),
            bool(chart_bytes),
        )
        return PresentedSignal(caption=caption, chart_bytes=chart_bytes)
