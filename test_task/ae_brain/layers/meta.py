"""Meta-model stacker (smart ensemble).

Supports:
* **Legacy 3-class** multinomial meta (SHORT / SKIP / LONG) for ablations.
* **Two-stage** meta: TRADE vs SKIP, then LONG vs SHORT on trade rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.meta")

_ARTIFACT_LEGACY = "meta_model.joblib"
_ARTIFACT_TWO_STAGE = "meta_two_stage.joblib"

CLASS_SHORT, CLASS_SKIP, CLASS_LONG = 0, 1, 2

META_INPUT_NAMES: tuple[str, ...] = (
    "tabular_p_up",
    "sequence_p_continuation",
    "sequence_trend_sign",
    "rl_target_exposure",
    "regime_trend",
    "regime_chop",
    "regime_highvol",
)


def build_meta_features(
    tab_p_up: float,
    seq_p_cont: float,
    seq_trend_sign: float,
    rl_exposure: float,
    regime_onehot: np.ndarray | list[float],
    *,
    layer_mask: dict[str, bool] | None = None,
) -> np.ndarray:
    """Assemble the meta input vector in canonical order."""
    mask = layer_mask or {}
    if not mask.get("tabular", True):
        tab_p_up = 0.5
    if not mask.get("sequence", True):
        seq_p_cont = 0.5
        seq_trend_sign = 0.0
    if not mask.get("rl", True):
        rl_exposure = 0.0

    reg = np.asarray(regime_onehot, dtype=float).reshape(-1)
    if reg.size < 3:
        reg = np.concatenate([reg, np.zeros(3 - reg.size)])
    vec = np.asarray(
        [tab_p_up, seq_p_cont, seq_trend_sign, rl_exposure, reg[0], reg[1], reg[2]],
        dtype=np.float32,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class MetaPrediction:
    """Legacy 3-class softmax output."""

    p_short: float
    p_skip: float
    p_long: float

    @property
    def confidence(self) -> float:
        return max(self.p_short, self.p_skip, self.p_long)


@dataclass(slots=True)
class TwoStagePrediction:
    """Two-stage meta output."""

    p_trade: float
    p_long_given_trade: float
    p_short_given_trade: float
    directional_class: int | None = None
    skip_reason: str | None = None

    @property
    def raw_confidence(self) -> float:
        if self.directional_class == CLASS_LONG:
            return self.p_trade * self.p_long_given_trade
        if self.directional_class == CLASS_SHORT:
            return self.p_trade * self.p_short_given_trade
        return self.p_trade * max(self.p_long_given_trade, self.p_short_given_trade)


def resolve_directional_class(
    p_short: float,
    p_long: float,
    *,
    threshold: float,
    margin: float = 0.0,
) -> tuple[int | None, str | None]:
    """Threshold-based directional signal with optional ambiguity margin.

    When both directional probabilities clear *threshold*:
    - if ``|p_long - p_short| < margin`` -> SKIP (directional ambiguity)
    - else argmax wins (no SHORT default)
    """
    long_ok = p_long > threshold
    short_ok = p_short > threshold
    if long_ok and short_ok:
        if abs(p_long - p_short) < margin:
            return None, "directional_ambiguity"
        return (CLASS_LONG, None) if p_long > p_short else (CLASS_SHORT, None)
    if long_ok:
        return CLASS_LONG, None
    if short_ok:
        return CLASS_SHORT, None
    return None, "below_direction_threshold"


def resolve_two_stage_direction(
    p_long_given_trade: float,
    p_short_given_trade: float,
    *,
    margin: float,
) -> tuple[int | None, str | None]:
    if abs(p_long_given_trade - p_short_given_trade) < margin:
        return None, "directional_ambiguity"
    return (CLASS_LONG, None) if p_long_given_trade > p_short_given_trade else (CLASS_SHORT, None)


class MetaModel:
    """Legacy 3-class multinomial meta-model."""

    name = "meta_legacy"

    def __init__(self, model_kind: str = "logreg") -> None:
        self._kind = model_kind
        self._scaler: Any = None
        self._clf: Any = None

    def _new_clf(self):
        if self._kind == "tree":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, class_weight="balanced")
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
        )

    def fit(self, F: np.ndarray, y: np.ndarray, *, train_end: int | None = None) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler

        F = np.asarray(F, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n = len(F)
        cut = train_end if train_end is not None else max(1, int(n * 0.8))
        scaler = StandardScaler().fit(F[:cut])
        clf = self._new_clf().fit(scaler.transform(F[:cut]), y[:cut])
        self._scaler, self._clf = scaler, clf

        yhat = clf.predict(scaler.transform(F[cut:]))
        ytrue = y[cut:]
        metrics = {
            "kind": "legacy_3class",
            "n": int(n),
            "n_val": int(len(ytrue)),
            "val_accuracy": float(accuracy_score(ytrue, yhat)) if len(ytrue) else float("nan"),
            "val_f1_macro": float(f1_score(ytrue, yhat, average="macro", zero_division=0))
            if len(ytrue)
            else float("nan"),
            "class_balance": {int(c): int((y == c).sum()) for c in (0, 1, 2)},
            "classes": [int(c) for c in clf.classes_],
        }
        log.info("meta.legacy.trained", **{k: v for k, v in metrics.items() if k != "class_balance"})
        return metrics

    def is_ready(self) -> bool:
        return self._clf is not None and self._scaler is not None

    def predict(self, meta_vec: np.ndarray) -> MetaPrediction:
        if not self.is_ready():
            return MetaPrediction(0.0, 1.0, 0.0)
        x = self._scaler.transform(np.asarray(meta_vec, dtype=np.float64).reshape(1, -1))
        proba = self._clf.predict_proba(x)[0]
        by_class = {int(c): float(p) for c, p in zip(self._clf.classes_, proba)}
        return MetaPrediction(
            by_class.get(CLASS_SHORT, 0.0),
            by_class.get(CLASS_SKIP, 0.0),
            by_class.get(CLASS_LONG, 0.0),
        )

    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"kind": self._kind, "scaler": self._scaler, "clf": self._clf},
            artifacts_dir / _ARTIFACT_LEGACY,
        )

    def load(self, artifacts_dir: Path) -> "MetaModel":
        import joblib

        path = Path(artifacts_dir) / _ARTIFACT_LEGACY
        if not path.exists():
            return self
        blob = joblib.load(path)
        self._kind = blob.get("kind", "logreg")
        self._scaler = blob["scaler"]
        self._clf = blob["clf"]
        return self


class TwoStageMetaModel:
    """Stage A: TRADE vs SKIP. Stage B: LONG vs SHORT on trade rows."""

    name = "meta_two_stage"

    def __init__(self, model_kind: str = "logreg") -> None:
        self._kind = model_kind
        self._scaler_a: Any = None
        self._scaler_d: Any = None
        self._action_clf: Any = None
        self._direction_clf: Any = None
        self._metrics: dict[str, Any] = {}

    def _binary_clf(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")

    def fit(self, F: np.ndarray, y: np.ndarray, *, train_end: int | None = None) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        F = np.asarray(F, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n = len(F)
        cut = train_end if train_end is not None else max(1, int(n * 0.8))

        y_action = (y != CLASS_SKIP).astype(int)
        scaler_a = StandardScaler().fit(F[:cut])
        action_clf = self._binary_clf().fit(scaler_a.transform(F[:cut]), y_action[:cut])

        trade_tr = y[:cut] != CLASS_SKIP
        y_dir = (y[:cut][trade_tr] == CLASS_LONG).astype(int)
        scaler_d = StandardScaler().fit(F[:cut][trade_tr])
        direction_clf = self._binary_clf().fit(scaler_d.transform(F[:cut][trade_tr]), y_dir)

        self._scaler_a, self._action_clf = scaler_a, action_clf
        self._scaler_d, self._direction_clf = scaler_d, direction_clf

        F_val = F[cut:]
        y_val = y[cut:]
        y_action_val = (y_val != CLASS_SKIP).astype(int)
        p_trade_val = action_clf.predict_proba(scaler_a.transform(F_val))[:, 1]
        trade_val = y_val != CLASS_SKIP
        p_long_val = direction_clf.predict_proba(scaler_d.transform(F_val[trade_val]))[:, 1] if trade_val.any() else np.array([])

        yhat_action = (p_trade_val >= 0.5).astype(int)
        dir_preds = []
        for i in range(len(F_val)):
            if p_trade_val[i] < 0.5:
                dir_preds.append(CLASS_SKIP)
                continue
            pl = direction_clf.predict_proba(scaler_d.transform(F_val[i : i + 1]))[0, 1]
            dir_preds.append(CLASS_LONG if pl >= 0.5 else CLASS_SHORT)
        yhat = np.asarray(dir_preds, dtype=int)

        action_auc = float("nan")
        try:
            if len(np.unique(y_action_val)) == 2:
                action_auc = float(roc_auc_score(y_action_val, p_trade_val))
        except Exception:
            pass

        self._metrics = {
            "kind": "two_stage",
            "n": int(n),
            "n_val": int(len(y_val)),
            "val_action_auc": action_auc,
            "val_action_accuracy": float(accuracy_score(y_action_val, yhat_action)) if len(y_val) else float("nan"),
            "val_directional_f1_macro": float(f1_score(y_val, yhat, average="macro", zero_division=0))
            if len(y_val)
            else float("nan"),
            "class_balance": {int(c): int((y == c).sum()) for c in (0, 1, 2)},
            "predicted_class_distribution_val": {
                "SHORT": int((yhat == CLASS_SHORT).sum()),
                "SKIP": int((yhat == CLASS_SKIP).sum()),
                "LONG": int((yhat == CLASS_LONG).sum()),
            },
        }
        log.info("meta.two_stage.trained", **{k: v for k, v in self._metrics.items() if k != "class_balance"})
        return self._metrics

    def is_ready(self) -> bool:
        return self._action_clf is not None and self._direction_clf is not None

    def predict(
        self,
        meta_vec: np.ndarray,
        *,
        trade_threshold: float = 0.45,
        direction_margin: float = 0.05,
    ) -> TwoStagePrediction:
        if not self.is_ready():
            return TwoStagePrediction(0.0, 0.5, 0.5, None, "meta_not_ready")
        x = np.asarray(meta_vec, dtype=np.float64).reshape(1, -1)
        p_trade = float(self._action_clf.predict_proba(self._scaler_a.transform(x))[0, 1])
        p_long = float(self._direction_clf.predict_proba(self._scaler_d.transform(x))[0, 1])
        p_short = 1.0 - p_long
        if p_trade < trade_threshold:
            return TwoStagePrediction(p_trade, p_long, p_short, None, "below_trade_threshold")
        directional_class, reason = resolve_two_stage_direction(p_long, p_short, margin=direction_margin)
        return TwoStagePrediction(p_trade, p_long, p_short, directional_class, reason)

    def predict_batch(self, F: np.ndarray, *, trade_threshold: float, direction_margin: float) -> list[TwoStagePrediction]:
        return [self.predict(row, trade_threshold=trade_threshold, direction_margin=direction_margin) for row in F]

    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self._kind,
                "scaler_a": self._scaler_a,
                "scaler_d": self._scaler_d,
                "action_clf": self._action_clf,
                "direction_clf": self._direction_clf,
                "metrics": self._metrics,
            },
            artifacts_dir / _ARTIFACT_TWO_STAGE,
        )

    def load(self, artifacts_dir: Path) -> "TwoStageMetaModel":
        import joblib

        path = Path(artifacts_dir) / _ARTIFACT_TWO_STAGE
        if not path.exists():
            return self
        blob = joblib.load(path)
        self._kind = blob.get("kind", "logreg")
        self._scaler_a = blob["scaler_a"]
        self._scaler_d = blob["scaler_d"]
        self._action_clf = blob["action_clf"]
        self._direction_clf = blob["direction_clf"]
        self._metrics = blob.get("metrics", {})
        return self


def load_meta_model(artifacts_dir: Path, *, prefer_two_stage: bool = True) -> MetaModel | TwoStageMetaModel | None:
    """Load the best available meta model artifact."""
    artifacts_dir = Path(artifacts_dir)
    if prefer_two_stage:
        two = TwoStageMetaModel().load(artifacts_dir)
        if two.is_ready():
            return two
    legacy = MetaModel().load(artifacts_dir)
    if legacy.is_ready():
        return legacy
    return None
