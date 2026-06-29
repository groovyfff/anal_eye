"""Unsupervised market-regime detector (Gaussian Mixture).

Classifies each candle into one of ``n_regimes`` regimes from a compact set of
volatility + momentum features, then exposes the regime as a *one-hot* vector
that is injected into the tabular and sequence models.

Semantic labels (for ``n_regimes == 3``), assigned deterministically after the
mixture is fit so the one-hot columns are stable across retrains:

* ``0 = trend``    - low/moderate volatility **with** directional strength (ADX),
* ``1 = chop``     - low volatility, weak directional strength (mean reversion),
* ``2 = highvol``  - high realized volatility (risk-off / breakout regime).

The model is intentionally lightweight (a scaler + a small full-covariance GMM)
so it adds negligible latency to the feature path and serializes to a single
``regime_model.joblib`` artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Compact, well-conditioned descriptors of "what kind of market is this".
# Every name here MUST be a *base* feature in the canonical schema (never a
# regime feature itself - that would be circular).
REGIME_INPUT_FEATURES: tuple[str, ...] = (
    "realized_vol_30",
    "atr_pct",
    "vol_of_vol",
    "bb_width",
    "adx_14",
    "rsi_14",
    "macd_hist",
    "roc_10",
    "ret_5",
    "hurst_exponent",
)

# Volatility descriptors used to rank regimes from calm -> turbulent.
_VOL_KEYS = ("realized_vol_30", "atr_pct", "vol_of_vol", "bb_width")

_ARTIFACT = "regime_model.joblib"


class RegimeModel:
    """Gaussian-Mixture market-regime classifier with stable label ordering."""

    def __init__(
        self,
        n_regimes: int = 3,
        features: Sequence[str] = REGIME_INPUT_FEATURES,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = int(n_regimes)
        self.features = tuple(features)
        self.random_state = random_state
        self._scaler = None
        self._gmm = None
        # Maps raw GMM component index -> semantic regime index (0..n-1).
        self._order: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    def _matrix(self, feat_df: pd.DataFrame) -> np.ndarray:
        cols = [
            feat_df[f].to_numpy(float) if f in feat_df else np.zeros(len(feat_df))
            for f in self.features
        ]
        X = np.column_stack(cols).astype(float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _semantic_order(self, X: np.ndarray, raw_labels: np.ndarray) -> dict[int, int]:
        """Map raw cluster ids -> stable semantic ids (trend/chop/highvol)."""
        df = pd.DataFrame(X, columns=list(self.features))
        df["_label"] = raw_labels
        means = df.groupby("_label").mean(numeric_only=True)
        present = list(means.index)

        vol_cols = [c for c in _VOL_KEYS if c in means.columns]
        vol_score = means[vol_cols].mean(axis=1) if vol_cols else means.mean(axis=1)

        if self.n_regimes != 3 or len(present) < 3:
            # Generic fallback: 0 = calmest ... n-1 = most turbulent.
            ordered = list(vol_score.sort_values().index)
            return {int(raw): pos for pos, raw in enumerate(ordered)}

        highvol = int(vol_score.idxmax())
        rest = [int(l) for l in present if int(l) != highvol]
        if "adx_14" in means.columns:
            trend = int(max(rest, key=lambda l: means.loc[l, "adx_14"]))
        else:
            trend = rest[0]
        chop = int([l for l in rest if l != trend][0])
        return {trend: 0, chop: 1, highvol: 2}

    # ------------------------------------------------------------------ #
    def fit(self, feat_df: pd.DataFrame) -> "RegimeModel":
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        X = self._matrix(feat_df)
        scaler = StandardScaler().fit(X)
        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_init=4,
            max_iter=200,
            reg_covar=1e-4,
        ).fit(scaler.transform(X))

        raw_labels = gmm.predict(scaler.transform(X))
        self._scaler, self._gmm = scaler, gmm
        self._order = self._semantic_order(X, raw_labels)
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        if not self.is_ready():
            return np.zeros(len(feat_df), dtype=int)
        raw = self._gmm.predict(self._scaler.transform(self._matrix(feat_df)))
        return np.array([self._order.get(int(r), int(r)) for r in raw], dtype=int)

    def one_hot(self, labels: np.ndarray) -> np.ndarray:
        labels = np.clip(np.asarray(labels, dtype=int), 0, self.n_regimes - 1)
        oh = np.zeros((len(labels), self.n_regimes), dtype=float)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    def predict_one_hot(self, feat_df: pd.DataFrame) -> np.ndarray:
        return self.one_hot(self.predict(feat_df))

    def is_ready(self) -> bool:
        return self._gmm is not None and self._scaler is not None

    # ------------------------------------------------------------------ #
    def save(self, artifacts_dir: Path) -> None:
        import joblib

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self._scaler,
                "gmm": self._gmm,
                "order": self._order,
                "features": list(self.features),
                "n_regimes": self.n_regimes,
            },
            artifacts_dir / _ARTIFACT,
        )

    def load(self, artifacts_dir: Path) -> "RegimeModel":
        import joblib

        path = Path(artifacts_dir) / _ARTIFACT
        if not path.exists():
            return self
        blob = joblib.load(path)
        self._scaler = blob["scaler"]
        self._gmm = blob["gmm"]
        self._order = {int(k): int(v) for k, v in blob["order"].items()}
        self.features = tuple(blob["features"])
        self.n_regimes = int(blob["n_regimes"])
        return self

    @classmethod
    def try_load(cls, artifacts_dir: Path) -> "RegimeModel | None":
        path = Path(artifacts_dir) / _ARTIFACT
        if not path.exists():
            return None
        return cls().load(artifacts_dir)
