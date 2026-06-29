"""Abstract base class for predictive layers."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class BasePredictor(abc.ABC):
    """Common lifecycle for a loadable, CPU/GPU inference component."""

    name: str = "base"

    @abc.abstractmethod
    def load(self, artifacts_dir: Path) -> None:
        """Load trained weights/artifacts from disk into memory."""

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return True if the layer can serve predictions."""

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous, thread/process-pool-safe prediction entrypoint."""
