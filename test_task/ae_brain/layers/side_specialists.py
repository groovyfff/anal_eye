"""Independent LONG/SHORT specialist binary models.

Each specialist predicts P(profitable setup | side) using EV-aware labels.
Trained and calibrated separately with no shared decision head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.layers.meta import META_INPUT_NAMES, build_meta_features
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.side_specialists")

_ARTIFACT_LONG = "long_specialist_model.joblib"
_ARTIFACT_SHORT = "short_specialist_model.joblib"


@dataclass(slots=True)
class SideSpecialistPrediction:
    side: str
    p_profitable_raw: float
    p_profitable_calibrated: float
    ev_usd: float
    confidence_adjusted_ev: float
    prob_tp: float
    prob_sl: float
    sizing_ok: bool
    publishable: bool
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SideSpecialistModel:
    """Binary classifier: profitable {LONG|SHORT} setup vs not."""

    def __init__(self, side: str, model_kind: str = "logreg") -> None:
        self.side = side.upper()
        self._kind = model_kind
        self._scaler: Any = None
        self._clf: Any = None
        self._metrics: dict[str, Any] = {}
        self._feature_names: list[str] | None = None

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def _new_clf(
        self,
        class_weight: float | None = None,
        scale_pos_weight: float | None = None,
    ):
        if self._kind == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
            params = {
                "objective": "binary",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "verbosity": -1,
            }
            if scale_pos_weight is not None and scale_pos_weight > 0:
                # scale_pos_weight and class_weight are mutually exclusive in LightGBM;
                # prefer scale_pos_weight when explicitly requested (XGBoost-style).
                params["scale_pos_weight"] = float(scale_pos_weight)
            elif class_weight is not None:
                params["class_weight"] = {0: 1.0, 1: float(class_weight)}
            return lgb.LGBMClassifier(**params)

        from sklearn.linear_model import LogisticRegression

        cw: Any = "balanced"
        if class_weight is not None:
            cw = {0: 1.0, 1: float(class_weight)}
        return LogisticRegression(C=0.5, max_iter=2000, class_weight=cw)

    def fit(
        self,
        F: np.ndarray,
        y: np.ndarray,
        *,
        train_end: int,
        class_weight: float | None = None,
        scale_pos_weight: float | None = None,
        train_idx: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        self._feature_names = feature_names
        F = np.asarray(F, dtype=np.float64)
        y = np.asarray(y, dtype=int).reshape(-1)
        n = len(F)
        # Optional balanced train subset (chronological positions only, no future
        # leakage): when train_idx is provided it replaces the [:cut] slice used
        # for fitting. The validation/calibration slice (F[cut:]) is unaffected.
        if train_idx is not None:
            tr = np.asarray(train_idx, dtype=int)
            tr = tr[(tr >= 0) & (tr < min(train_end, n))]
        else:
            tr = np.arange(min(train_end, n))
        cut = max(1, min(train_end, n - 1))
        scaler = StandardScaler().fit(F[tr])
        clf = self._new_clf(class_weight, scale_pos_weight).fit(
            scaler.transform(F[tr]), y[tr]
        )
        self._scaler, self._clf = scaler, clf

        val_x = scaler.transform(F[cut:])
        val_y = y[cut:]
        if hasattr(clf, "predict_proba"):
            val_proba = clf.predict_proba(val_x)[:, 1] if len(val_y) else np.array([])
        else:
            val_proba = np.array([])
        pos_rate = float(y[tr].mean()) if tr.size else 0.0
        val_pos = int(val_y.sum()) if len(val_y) else 0
        metrics: dict[str, Any] = {
            "side": self.side,
            "model_kind": self._kind,
            "n_train": int(tr.size),
            "n_val": int(len(val_y)),
            "train_positive_rate": pos_rate,
            "train_positive_count": int(y[tr].sum()),
            "val_positive_count": val_pos,
            "val_positive_rate": float(val_y.mean()) if len(val_y) else 0.0,
            "val_auc": None,
            "class_weight": float(class_weight) if class_weight is not None else None,
            "scale_pos_weight": float(scale_pos_weight) if scale_pos_weight is not None else None,
            "balanced_train_samples": train_idx is not None,
        }
        if len(val_y) >= 10 and len(np.unique(val_y)) >= 2 and val_proba.size:
            metrics["val_auc"] = float(roc_auc_score(val_y, val_proba))
        self._metrics = metrics
        log.info("side_specialist.trained", **metrics)
        return metrics

    @property
    def artifact_name(self) -> str:
        return _ARTIFACT_LONG if self.side == "LONG" else _ARTIFACT_SHORT

    def is_ready(self) -> bool:
        return self._clf is not None and self._scaler is not None

    def predict_raw(self, meta_vec: np.ndarray) -> float:
        if not self.is_ready():
            return 0.0
        x = self._scaler.transform(np.asarray(meta_vec, dtype=np.float64).reshape(1, -1))
        proba = self._clf.predict_proba(x)[0]
        # class 1 = profitable setup
        classes = list(self._clf.classes_)
        idx = classes.index(1) if 1 in classes else -1
        return float(proba[idx]) if idx >= 0 else 0.0

    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "side": self.side,
                "kind": self._kind,
                "scaler": self._scaler,
                "clf": self._clf,
                "metrics": self._metrics,
                "feature_names": self._feature_names or list(META_INPUT_NAMES),
            },
            artifacts_dir / self.artifact_name,
        )

    def load(self, artifacts_dir: Path) -> "SideSpecialistModel":
        import joblib

        path = Path(artifacts_dir) / self.artifact_name
        if not path.exists():
            return self
        blob = joblib.load(path)
        self.side = blob.get("side", self.side)
        self._kind = blob.get("kind", "logreg")
        self._scaler = blob["scaler"]
        self._clf = blob["clf"]
        self._metrics = blob.get("metrics", {})
        return self


@dataclass(slots=True)
class SideSpecialistsBundle:
    long_model: SideSpecialistModel
    short_model: SideSpecialistModel

    def is_ready(self) -> bool:
        return self.long_model.is_ready() and self.short_model.is_ready()


def resolve_side_specialist_decision(
    long_prob: float,
    short_prob: float,
    *,
    publish_threshold: float = 0.70,
) -> tuple[str, str | None]:
    """Production side_specialists direction from calibrated specialist probabilities.

    LONG when long_prob >= threshold and long_prob > short_prob.
    SHORT when short_prob >= threshold and short_prob > long_prob.
    SKIP otherwise (including equal qualifying probabilities).
    """
    if long_prob >= publish_threshold and long_prob > short_prob:
        return "LONG", "long_prob_ge_threshold_and_gt_short"
    if short_prob >= publish_threshold and short_prob > long_prob:
        return "SHORT", "short_prob_ge_threshold_and_gt_long"
    return "SKIP", None


def load_side_specialists(artifacts_dir: Path) -> SideSpecialistsBundle:
    return SideSpecialistsBundle(
        long_model=SideSpecialistModel("LONG").load(artifacts_dir),
        short_model=SideSpecialistModel("SHORT").load(artifacts_dir),
    )


def build_specialist_features(
    tab_p_up: float,
    seq_p_cont: float,
    seq_trend_sign: float,
    rl_exposure: float,
    regime_onehot: np.ndarray | list[float],
    *,
    layer_mask: dict[str, bool] | None = None,
) -> np.ndarray:
    return build_meta_features(
        tab_p_up, seq_p_cont, seq_trend_sign, rl_exposure, regime_onehot, layer_mask=layer_mask
    )
