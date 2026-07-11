"""Per-side confidence calibration (validation-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_ARTIFACT = "confidence_calibrator.joblib"
_ARTIFACT_LONG = "confidence_calibrator_long.joblib"
_ARTIFACT_SHORT = "confidence_calibrator_short.joblib"


@dataclass(slots=True)
class CalibrationReport:
    method: str
    n_samples: int
    brier_raw: float | None
    brier_calibrated: float | None
    raw_confidence: dict[str, float]
    calibrated_confidence: dict[str, float]
    publishable_counts: dict[str, int]
    side: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "brier_raw": self.brier_raw,
            "brier_calibrated": self.brier_calibrated,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "publishable_counts": self.publishable_counts,
            "side": self.side,
        }


class ConfidenceCalibrator:
    """Maps raw trade confidence scores to P(profitable / EV-positive) on [0, 1]."""

    def __init__(
        self,
        method: Literal["isotonic", "sigmoid"] = "isotonic",
        *,
        artifact_name: str = _ARTIFACT,
        side: str | None = None,
    ) -> None:
        self._method = method
        self._artifact_name = artifact_name
        self._side = side
        self._model: IsotonicRegression | LogisticRegression | None = None
        self._report: CalibrationReport | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def report(self) -> CalibrationReport | None:
        return self._report

    def fit(self, raw_scores: np.ndarray, y_profitable: np.ndarray) -> CalibrationReport:
        raw = np.clip(np.asarray(raw_scores, dtype=float).reshape(-1), 0.0, 1.0)
        y = np.asarray(y_profitable, dtype=int).reshape(-1)
        if raw.size < 20 or len(np.unique(y)) < 2:
            self._model = None
            self._report = CalibrationReport(
                method=self._method,
                n_samples=int(raw.size),
                brier_raw=None,
                brier_calibrated=None,
                raw_confidence=_dist_stats(raw),
                calibrated_confidence={},
                publishable_counts=_publishable_counts(raw),
                side=self._side,
            )
            return self._report

        brier_raw = float(brier_score_loss(y, raw))
        if self._method == "isotonic":
            model: Any = IsotonicRegression(out_of_bounds="clip")
            model.fit(raw, y)
            calibrated = np.clip(model.predict(raw), 0.0, 1.0)
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(raw.reshape(-1, 1), y)
            calibrated = np.clip(model.predict_proba(raw.reshape(-1, 1))[:, 1], 0.0, 1.0)
        self._model = model
        brier_cal = float(brier_score_loss(y, calibrated))
        self._report = CalibrationReport(
            method=self._method,
            n_samples=int(raw.size),
            brier_raw=brier_raw,
            brier_calibrated=brier_cal,
            raw_confidence=_dist_stats(raw),
            calibrated_confidence=_dist_stats(calibrated),
            publishable_counts=_publishable_counts(calibrated),
            side=self._side,
        )
        return self._report

    def calibrate(self, raw_score: float) -> float:
        if not self.is_ready:
            return float(np.clip(raw_score, 0.0, 1.0))
        raw = float(np.clip(raw_score, 0.0, 1.0))
        if isinstance(self._model, IsotonicRegression):
            return float(np.clip(self._model.predict([raw])[0], 0.0, 1.0))
        return float(np.clip(self._model.predict_proba([[raw]])[0, 1], 0.0, 1.0))

    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self._method,
                "model": self._model,
                "report": self._report.to_dict() if self._report else None,
                "side": self._side,
            },
            artifacts_dir / self._artifact_name,
        )

    def load(self, artifacts_dir: Path) -> "ConfidenceCalibrator":
        import joblib

        path = Path(artifacts_dir) / self._artifact_name
        if not path.exists():
            return self
        blob = joblib.load(path)
        self._method = blob.get("method", "isotonic")
        self._model = blob.get("model")
        rep = blob.get("report")
        if rep:
            self._report = CalibrationReport(**rep)
        return self


class SideCalibrators:
    """Separate LONG and SHORT confidence calibrators."""

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic") -> None:
        self.long = ConfidenceCalibrator(method, artifact_name=_ARTIFACT_LONG, side="LONG")
        self.short = ConfidenceCalibrator(method, artifact_name=_ARTIFACT_SHORT, side="SHORT")
        self._legacy = ConfidenceCalibrator(method, artifact_name=_ARTIFACT)

    def calibrate(self, side: str, raw: float) -> float:
        if side == "LONG":
            return self.long.calibrate(raw)
        if side == "SHORT":
            return self.short.calibrate(raw)
        return self._legacy.calibrate(raw)

    def is_ready(self, side: str) -> bool:
        if side == "LONG":
            return self.long.is_ready
        if side == "SHORT":
            return self.short.is_ready
        return self._legacy.is_ready

    def save(self, artifacts_dir: Path) -> None:
        self.long.save(artifacts_dir)
        self.short.save(artifacts_dir)

    def load(self, artifacts_dir: Path) -> "SideCalibrators":
        self.long.load(artifacts_dir)
        self.short.load(artifacts_dir)
        self._legacy.load(artifacts_dir)
        return self

    def reports(self) -> dict[str, Any]:
        return {
            "LONG": self.long.report.to_dict() if self.long.report else {},
            "SHORT": self.short.report.to_dict() if self.short.report else {},
        }


def _dist_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0.0}
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _publishable_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for thr in (0.50, 0.60, 0.70, 0.80):
        out[f"ge_{thr:.2f}"] = int((values >= thr).sum())
    return out
