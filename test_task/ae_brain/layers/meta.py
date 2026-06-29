"""Meta-model stacker (smart ensemble).

A lightweight multinomial classifier that takes the *outputs* of the three base
layers plus the current market regime and emits the final absolute decision:

    inputs  = [tabular_p_up,
               sequence_p_continuation,
               sequence_trend_sign,
               rl_target_exposure,
               regime_trend, regime_chop, regime_highvol]
    output  = threshold P(LONG) / P(SHORT) over {0: SHORT, 1: SKIP, 2: LONG}
              (SKIP is not argmax-competed; see ``resolve_directional_class``)

This *replaces* the static/heuristic EV gate's directional decision (the risk
engine still prices TP/SL and sizes the position). It is trained on realized
3-class first-passage outcomes (see ``dataset.directional_barrier_labels``), so
it learns *when the base models actually agree with a profitable outcome* rather
than trusting a fixed weighted blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.meta")

_ARTIFACT = "meta_model.joblib"

# Class id <-> semantic decision (matches directional_barrier_labels).
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
) -> np.ndarray:
    """Assemble the meta input vector in canonical order."""
    reg = np.asarray(regime_onehot, dtype=float).reshape(-1)
    if reg.size < 3:
        reg = np.concatenate([reg, np.zeros(3 - reg.size)])
    vec = np.asarray(
        [tab_p_up, seq_p_cont, seq_trend_sign, rl_exposure, reg[0], reg[1], reg[2]],
        dtype=np.float32,
    )
    # A NaN/inf from any base layer must never crash the stacker; map to neutral.
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class MetaPrediction:
    p_short: float
    p_skip: float
    p_long: float

    @property
    def confidence(self) -> float:
        return max(self.p_short, self.p_skip, self.p_long)


def resolve_directional_class(
    p_short: float,
    p_long: float,
    *,
    threshold: float,
) -> int | None:
    """Threshold-based directional signal; ignores ``p_skip`` (class-imbalance safe).

    Returns ``CLASS_LONG``, ``CLASS_SHORT``, or ``None`` when neither directional
    probability clears *threshold*. When both clear it, the higher probability wins.
    """
    long_ok = p_long > threshold
    short_ok = p_short > threshold
    if long_ok and short_ok:
        return CLASS_LONG if p_long >= p_short else CLASS_SHORT
    if long_ok:
        return CLASS_LONG
    if short_ok:
        return CLASS_SHORT
    return None


class MetaModel:
    name = "meta"

    def __init__(self, model_kind: str = "logreg") -> None:
        self._kind = model_kind
        self._scaler: Any = None
        self._clf: Any = None

    # ------------------------------------------------------------------ #
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

    def fit(self, F: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler

        F = np.asarray(F, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n = len(F)
        # Time-ordered split for an honest meta validation.
        cut = max(1, int(n * 0.8))
        scaler = StandardScaler().fit(F[:cut])
        clf = self._new_clf().fit(scaler.transform(F[:cut]), y[:cut])
        self._scaler, self._clf = scaler, clf

        yhat = clf.predict(scaler.transform(F[cut:]))
        ytrue = y[cut:]
        metrics = {
            "n": int(n),
            "n_val": int(len(ytrue)),
            "val_accuracy": float(accuracy_score(ytrue, yhat)) if len(ytrue) else float("nan"),
            "val_f1_macro": float(f1_score(ytrue, yhat, average="macro", zero_division=0))
            if len(ytrue)
            else float("nan"),
            "class_balance": {int(c): int((y == c).sum()) for c in (0, 1, 2)},
            "classes": [int(c) for c in clf.classes_],
        }
        log.info("meta.trained", **{k: v for k, v in metrics.items() if k != "class_balance"})
        return metrics

    # ------------------------------------------------------------------ #
    def is_ready(self) -> bool:
        return self._clf is not None and self._scaler is not None

    def predict(self, meta_vec: np.ndarray) -> MetaPrediction:
        if not self.is_ready():
            # Neutral fallback -> SKIP (no directional confidence).
            return MetaPrediction(0.0, 1.0, 0.0)
        x = self._scaler.transform(np.asarray(meta_vec, dtype=np.float64).reshape(1, -1))
        proba = self._clf.predict_proba(x)[0]
        by_class = {int(c): float(p) for c, p in zip(self._clf.classes_, proba)}
        p_short = by_class.get(CLASS_SHORT, 0.0)
        p_skip = by_class.get(CLASS_SKIP, 0.0)
        p_long = by_class.get(CLASS_LONG, 0.0)

        log.info(
            "meta.predict",
            p_short=round(p_short, 6),
            p_skip=round(p_skip, 6),
            p_long=round(p_long, 6),
        )
        return MetaPrediction(p_short, p_skip, p_long)

    # ------------------------------------------------------------------ #
    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"kind": self._kind, "scaler": self._scaler, "clf": self._clf},
            artifacts_dir / _ARTIFACT,
        )
        log.info("meta.saved", dir=str(artifacts_dir))

    def load(self, artifacts_dir: Path) -> "MetaModel":
        import joblib

        path = Path(artifacts_dir) / _ARTIFACT
        if not path.exists():
            log.warning("meta.no_model", dir=str(artifacts_dir))
            return self
        blob = joblib.load(path)
        self._kind = blob.get("kind", "logreg")
        self._scaler = blob["scaler"]
        self._clf = blob["clf"]
        log.info("meta.loaded", ready=self.is_ready())
        return self
