"""Layer 2 - Sequence / Time-Series predictor.

Consumes a window of OHLCV(+derived) candles (>= 30, default 48) and outputs:
  * ``p_continuation`` - calibrated probability of trend continuation,
  * ``trend_sign``     - direction of the prevailing trend in [-1, 1].

Runs in fp16 on the P100s. If an exported ONNX graph is present and ONNX is
preferred, inference uses ONNXRuntime-GPU; otherwise the native torch module is
used. Both paths are synchronous and release the GIL during the heavy matmul,
so they are dispatched from the async loop via a ThreadPoolExecutor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ae_brain.config import GPUConfig, ModelConfig
from ae_brain.layers.base import BasePredictor
from ae_brain.utils.gpu import DeviceRouter
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.sequence")

_WEIGHTS_FILE = "sequence_model.pt"
_ONNX_FILE = "sequence_model.onnx"
_NORM_FILE = "sequence_norm.npz"

# Channels fed to the sequence model (a compact OHLCV+flow view) plus the
# per-bar market-regime one-hot (filled with zeros when no regime model is
# attached, so short/legacy windows still produce a valid signal).
SEQ_CHANNELS = (
    "open", "high", "low", "close", "volume",
    "taker_buy_volume", "open_interest", "funding_rate",
    "regime_trend", "regime_chop", "regime_highvol",
)


@dataclass(slots=True)
class SequencePrediction:
    p_continuation: float
    trend_sign: float


class SequencePredictor(BasePredictor):
    name = "sequence"

    def __init__(self, model_cfg: ModelConfig, gpu_cfg: GPUConfig) -> None:
        self._cfg = model_cfg
        self._gpu = gpu_cfg
        self._router = DeviceRouter(gpu_cfg)
        self._module: Any = None
        self._onnx_session: Any = None
        self._device = "cpu"
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def window(self) -> int:
        return self._cfg.sequence_window

    @property
    def n_channels(self) -> int:
        return len(SEQ_CHANNELS)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _to_window_array(self, candles) -> np.ndarray:
        """Extract, order and pad/truncate the channel matrix (T, C)."""
        import pandas as pd

        df = candles if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)
        cols = {}
        for ch in SEQ_CHANNELS:
            cols[ch] = df[ch].to_numpy(float) if ch in df else np.zeros(len(df))
        mat = np.column_stack([cols[ch] for ch in SEQ_CHANNELS])
        if len(mat) < self.window:
            pad = np.repeat(mat[:1], self.window - len(mat), axis=0)
            mat = np.vstack([pad, mat])
        return mat[-self.window :]

    def _normalize(self, mat: np.ndarray) -> np.ndarray:
        if self._mean is not None and self._std is not None:
            return (mat - self._mean) / np.where(self._std == 0, 1.0, self._std)
        # Per-window standardization fallback (robust to scale drift).
        mu = mat.mean(axis=0, keepdims=True)
        sd = mat.std(axis=0, keepdims=True)
        return (mat - mu) / np.where(sd == 0, 1.0, sd)

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._mean, self._std = mean, std

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def build_module(self) -> Any:
        from ae_brain.layers.nets import build_network

        return build_network(self._cfg.sequence_backend, self.n_channels, self.window)

    def load(self, artifacts_dir: Path) -> None:
        norm_path = artifacts_dir / _NORM_FILE
        if norm_path.exists():
            data = np.load(norm_path)
            self._mean, self._std = data["mean"], data["std"]

        onnx_path = artifacts_dir / _ONNX_FILE
        if self._gpu.prefer_onnx and onnx_path.exists():
            self._load_onnx(onnx_path)
            return

        weights_path = artifacts_dir / _WEIGHTS_FILE
        if weights_path.exists():
            self._load_torch(weights_path)
        else:
            log.warning("sequence.no_weights", dir=str(artifacts_dir))

    def _load_torch(self, weights_path: Path) -> None:
        import torch

        from ae_brain.utils.gpu import to_inference

        module = self.build_module()
        state = torch.load(weights_path, map_location="cpu")
        module.load_state_dict(state)
        self._device = self._router.next_device()
        self._module = to_inference(module, self._device, self._gpu)
        log.info("sequence.torch.loaded", device=self._device, fp16=self._gpu.use_fp16)

    def _load_onnx(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._router.cuda_available()
            else ["CPUExecutionProvider"]
        )
        self._onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        log.info("sequence.onnx.loaded", providers=providers)

    def export_onnx(self, artifacts_dir: Path) -> Path:
        """Export the loaded torch module to ONNX (fp16) for P100 serving."""
        import torch

        if self._module is None:
            raise RuntimeError("no torch module loaded to export")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = artifacts_dir / _ONNX_FILE
        dummy = torch.randn(1, self.window, self.n_channels, device=self._device)
        if self._gpu.use_fp16 and self._device.startswith("cuda"):
            dummy = dummy.half()
        torch.onnx.export(
            self._module,
            dummy,
            str(out),
            input_names=["window"],
            output_names=["cont_logit", "trend_sign"],
            dynamic_axes={"window": {0: "batch"}},
            opset_version=17,
        )
        log.info("sequence.onnx.exported", path=str(out))
        return out

    def is_ready(self) -> bool:
        return self._module is not None or self._onnx_session is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-x)))

    def predict(self, candles) -> SequencePrediction:
        mat = self._normalize(self._to_window_array(candles)).astype(np.float32)
        if not self.is_ready():
            return SequencePrediction(p_continuation=0.5, trend_sign=0.0)

        if self._onnx_session is not None:
            cont_logit, sign = self._predict_onnx(mat)
        else:
            cont_logit, sign = self._predict_torch(mat)
        return SequencePrediction(
            p_continuation=self._sigmoid(cont_logit),
            trend_sign=float(np.clip(sign, -1.0, 1.0)),
        )

    def _predict_torch(self, mat: np.ndarray) -> tuple[float, float]:
        import torch

        x = torch.from_numpy(mat).unsqueeze(0).to(self._device)
        if self._gpu.use_fp16 and self._device.startswith("cuda"):
            x = x.half()
        with torch.no_grad():
            cont_logit, sign = self._module(x)
        return float(cont_logit.float().cpu().item()), float(sign.float().cpu().item())

    def _predict_onnx(self, mat: np.ndarray) -> tuple[float, float]:
        x = mat[None, ...]
        # ONNX fp16 graph expects fp16 input.
        in_type = self._onnx_session.get_inputs()[0].type
        if "float16" in in_type:
            x = x.astype(np.float16)
        cont_logit, sign = self._onnx_session.run(None, {"window": x})
        return float(np.asarray(cont_logit).reshape(-1)[0]), float(np.asarray(sign).reshape(-1)[0])
