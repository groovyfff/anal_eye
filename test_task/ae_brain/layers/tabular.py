"""Layer 1 - Tabular Predictor with strict probability calibration.

Trains a gradient-boosted tree (LightGBM / XGBoost / CatBoost) on the ~60
canonical features, then wraps it in a calibrator (Isotonic regression or Platt
/ sigmoid scaling) so that the emitted score is a *true* probability that can be
fed directly into the EV gate.

Calibration matters enormously here: the EV formula multiplies probabilities by
USD amounts, so a miscalibrated 0.7 that is really 0.55 directly produces
phantom positive-EV trades.

Inference is synchronous and GIL-light (numpy + C++ boosters), so it is safe to
dispatch from a ``ProcessPoolExecutor`` / ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.config import ModelConfig
from ae_brain.features.schema import FEATURE_NAMES
from ae_brain.layers.base import BasePredictor
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.tabular")

_MODEL_FILE = "tabular_model.joblib"
_CALIBRATOR_FILE = "tabular_calibrator.joblib"
# Authoritative list of features the (possibly SHAP-pruned) tabular model
# consumes, in order. Written by the production trainer; read here so inference
# selects exactly the kept columns from the full canonical feature vector.
_FEATURES_SCHEMA_FILE = "features_schema.json"


def _resolve_kept(kept_names: list[str]) -> np.ndarray:
    """Indices into the full FEATURE_NAMES vector for the kept feature subset."""
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    missing = [n for n in kept_names if n not in index]
    if missing:
        raise ValueError(f"kept features not in canonical schema: {missing}")
    return np.asarray([index[n] for n in kept_names], dtype=int)


@dataclass(slots=True)
class TabularPrediction:
    p_up: float  # calibrated P(price reaches +R target before -R)
    raw_score: float


class TabularPredictor(BasePredictor):
    name = "tabular"

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._calibrator: Any = None
        # Kept-feature subset (defaults to the full canonical schema).
        self._kept_names: list[str] = list(FEATURE_NAMES)
        self._kept_idx: np.ndarray = np.arange(len(FEATURE_NAMES))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _new_booster(self) -> Any:
        backend = self._cfg.tabular_backend
        if backend == "lightgbm":
            from lightgbm import LGBMClassifier

            # Tuned for noisy financial labels: low LR + many trees capped by
            # early stopping, shallow-ish leaves + strong L1/L2 + row/col
            # subsampling to fight overfitting on regime-specific noise.
            return LGBMClassifier(
                n_estimators=4000,
                learning_rate=0.02,
                num_leaves=48,
                max_depth=8,
                min_child_samples=60,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                objective="binary",
                n_jobs=-1,
                verbosity=-1,
            )
        if backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
            )
        if backend == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=600,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                verbose=False,
            )
        raise ValueError(f"unknown tabular backend {backend!r}")

    def _fit_booster(self, booster: Any, X_fit, y_fit, X_val=None, y_val=None) -> Any:
        """Fit one booster, using LightGBM early stopping when a val set exists."""
        has_val = (
            X_val is not None
            and len(X_val) > 0
            and len(np.unique(y_val)) >= 2  # AUC early-stopping needs both classes
        )
        if self._cfg.tabular_backend == "lightgbm" and has_val:
            from lightgbm import early_stopping, log_evaluation

            booster.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
            )
        else:
            booster.fit(X_fit, y_fit)
        return booster

    def _cv_auc(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> list[float]:
        """Walk-forward (TimeSeriesSplit) AUC to detect look-ahead/overfit.

        Each fold trains only on the *past* and validates on the immediate
        future, with early stopping on the validation fold.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        if len(X) < (n_splits + 1) * 50:
            return []
        aucs: list[float] = []
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr, va in tss.split(X):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            booster = self._fit_booster(self._new_booster(), X[tr], y[tr], X[va], y[va])
            p = booster.predict_proba(X[va])[:, 1]
            aucs.append(float(roc_auc_score(y[va], p)))
        return aucs

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        calib_size: float = 0.15,
        val_size: float = 0.15,
        n_splits: int = 4,
    ) -> dict[str, float]:
        """Fit booster + calibrator with walk-forward validation.

        Pipeline (all time-ordered, no shuffling -> no look-ahead bias):

        1. **TimeSeriesSplit CV** for an honest, cross-validated AUC estimate.
        2. **Final fit** on the oldest ``1 - val - calib`` slice, with
           LightGBM early stopping on the ``val`` slice (tunes tree count).
        3. **Calibration** on the most-recent ``calib`` slice (held out from the
           booster fit) so the emitted score is a true probability for the EV gate.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if feature_names is not None:
            # ``feature_names`` is the (possibly SHAP-pruned) kept subset, in the
            # same column order as ``X``. Record it so inference selects exactly
            # these columns from the full canonical vector.
            self._kept_names = list(feature_names)
            self._kept_idx = _resolve_kept(self._kept_names)
        if X.shape[1] != len(self._kept_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(self._kept_names)} kept feature names"
            )

        # 1) Cross-validated AUC (walk-forward).
        cv_aucs = self._cv_auc(X, y, n_splits)

        # 2) Time-ordered fit / val / calib partition.
        n = len(X)
        n_cal = max(1, int(n * calib_size))
        n_val = max(1, int(n * val_size))
        fit_end = max(1, n - n_cal - n_val)
        X_fit, y_fit = X[:fit_end], y[:fit_end]
        X_val, y_val = X[fit_end : fit_end + n_val], y[fit_end : fit_end + n_val]
        X_cal, y_cal = X[fit_end + n_val :], y[fit_end + n_val :]

        booster = self._fit_booster(self._new_booster(), X_fit, y_fit, X_val, y_val)

        method = "isotonic" if self._cfg.calibration_method == "isotonic" else "sigmoid"
        # Calibrate on the held-out recent fold against the already-fit booster.
        try:
            from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6

            calibrator = CalibratedClassifierCV(FrozenEstimator(booster), method=method)
        except ImportError:  # sklearn < 1.6
            calibrator = CalibratedClassifierCV(booster, method=method, cv="prefit")
        calibrator.fit(X_cal, y_cal)

        self._model = booster
        self._calibrator = calibrator

        p_cal = calibrator.predict_proba(X_cal)[:, 1]
        yhat = (p_cal >= 0.5).astype(int)
        best_iter = int(getattr(booster, "best_iteration_", 0) or 0)
        metrics = {
            "auc": float(roc_auc_score(y_cal, p_cal)),
            "cv_auc_mean": float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
            "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else float("nan"),
            "precision": float(precision_score(y_cal, yhat, zero_division=0)),
            "recall": float(recall_score(y_cal, yhat, zero_division=0)),
            "log_loss": float(log_loss(y_cal, p_cal, labels=[0, 1])),
            "brier": float(brier_score_loss(y_cal, p_cal)),
            "best_iteration": best_iter,
            "pos_rate": float(np.mean(y)),
            "n_fit": int(len(y_fit)),
            "n_val": int(len(y_val)),
            "n_cal": int(len(y_cal)),
        }
        log.info("tabular.trained", backend=self._cfg.tabular_backend, **metrics)
        return metrics

    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, artifacts_dir / _MODEL_FILE)
        joblib.dump(self._calibrator, artifacts_dir / _CALIBRATOR_FILE)
        log.info("tabular.saved", dir=str(artifacts_dir))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def load(self, artifacts_dir: Path) -> None:
        import joblib

        model_path = artifacts_dir / _MODEL_FILE
        calib_path = artifacts_dir / _CALIBRATOR_FILE
        if model_path.exists():
            self._model = joblib.load(model_path)
        if calib_path.exists():
            self._calibrator = joblib.load(calib_path)

        # Resolve the kept-feature subset (SHAP-pruned). Falls back to the full
        # canonical schema when no features_schema.json is present.
        schema_path = artifacts_dir / _FEATURES_SCHEMA_FILE
        if schema_path.exists():
            try:
                blob = json.loads(schema_path.read_text())
                kept = blob.get("kept_features") or list(FEATURE_NAMES)
                self._kept_names = list(kept)
                self._kept_idx = _resolve_kept(self._kept_names)
            except Exception as exc:  # pragma: no cover - tolerate bad schema
                log.warning("tabular.features_schema_failed", err=str(exc))
                self._kept_names = list(FEATURE_NAMES)
                self._kept_idx = np.arange(len(FEATURE_NAMES))
        log.info("tabular.loaded", ready=self.is_ready(), n_features=len(self._kept_names))

    def is_ready(self) -> bool:
        return self._calibrator is not None

    def predict(self, features: np.ndarray) -> TabularPrediction:
        """Return a calibrated up-probability for one feature vector.

        Falls back to a neutral 0.5 if no model is loaded (safe default that the
        EV gate will reject as non-edge).
        """
        full = np.asarray(features, dtype=np.float32).reshape(-1)
        # Select the kept (SHAP-pruned) columns from the full canonical vector.
        x = full[self._kept_idx].reshape(1, len(self._kept_names))
        if not self.is_ready():
            return TabularPrediction(p_up=0.5, raw_score=0.5)
        p = float(self._calibrator.predict_proba(x)[0, 1])
        raw = (
            float(self._model.predict_proba(x)[0, 1])
            if self._model is not None
            else p
        )
        return TabularPrediction(p_up=p, raw_score=raw)
