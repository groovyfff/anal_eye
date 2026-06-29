"""GPU / precision helpers targeting 4x Tesla P100 (Pascal, sm_60).

The P100 supports native fp16 storage/compute. We therefore:
  * pick devices round-robin across the 4 cards,
  * cast inference modules + inputs to fp16 when enabled,
  * enable cuDNN autotuner for fixed-shape sequence inference.

Torch is imported lazily so that pure-CPU components (feature engineering,
EV gate, fusion math) do not require a CUDA build to be installed.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterator

from ae_brain.config import GPUConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _import_torch():
    try:
        import torch  # noqa: WPS433 (runtime import is intentional)

        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for GPU/sequence inference. Install the CUDA "
            "build matching your P100 driver, e.g. torch==2.2.2+cu118."
        ) from exc


class DeviceRouter:
    """Round-robin allocator that spreads models across the available P100s."""

    def __init__(self, cfg: GPUConfig) -> None:
        self._cfg = cfg
        self._cycle: Iterator[int] = itertools.cycle(cfg.device_ids or [0])

    def cuda_available(self) -> bool:
        if not self._cfg.enabled:
            return False
        try:
            torch = _import_torch()
        except RuntimeError:
            return False
        return bool(torch.cuda.is_available())

    def next_device(self) -> str:
        """Return the next torch device string (round-robin across cards)."""
        if not self.cuda_available():
            return "cpu"
        return f"cuda:{next(self._cycle)}"

    @property
    def dtype(self):
        """Return the inference dtype (fp16 on P100 when enabled)."""
        torch = _import_torch()
        return torch.float16 if (self._cfg.use_fp16 and self.cuda_available()) else torch.float32


def to_inference(module: "torch.nn.Module", device: str, cfg: GPUConfig) -> "torch.nn.Module":
    """Move a module to ``device``, set eval mode and (optionally) fp16."""
    torch = _import_torch()
    module = module.to(device)
    module.eval()
    if cfg.use_fp16 and device.startswith("cuda"):
        module = module.half()
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    return module
