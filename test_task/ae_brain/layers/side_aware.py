"""Side-aware ensemble: per-side source selection and candidate generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.contracts import Decision, LayerProbabilities, Side
from ae_brain.layers.meta import (
    CLASS_LONG,
    CLASS_SHORT,
    MetaModel,
    MetaPrediction,
    TwoStageMetaModel,
    TwoStagePrediction,
    build_meta_features,
    resolve_directional_class,
    resolve_two_stage_direction,
)

_ARTIFACT = "side_aware_ensemble.json"

SOURCE_MODES = (
    "tabular_only",
    "no_meta",
    "two_stage_meta",
    "legacy_3class_meta",
)


@dataclass(slots=True)
class SideCandidate:
    side: str
    source: str
    raw_confidence: float
    calibrated_confidence: float
    ev_usd: float
    utility: float
    fused_score: float
    prob_tp: float
    prob_sl: float
    risk_approved: bool
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SideAwareConfig:
    long_source: str
    short_source: str
    selection_split: str = "validation"
    long_source_metrics: dict[str, Any] | None = None
    short_source_metrics: dict[str, Any] | None = None
    publish_confidence: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SideAwareConfig":
        return cls(
            long_source=data.get("long_source", "two_stage_meta"),
            short_source=data.get("short_source", "no_meta"),
            selection_split=data.get("selection_split", "validation"),
            long_source_metrics=data.get("long_source_metrics"),
            short_source_metrics=data.get("short_source_metrics"),
            publish_confidence=float(data.get("publish_confidence", 0.70)),
        )


def save_side_aware_config(config: SideAwareConfig, artifacts_dir: Path) -> Path:
    path = Path(artifacts_dir) / _ARTIFACT
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return path


def load_side_aware_config(artifacts_dir: Path) -> SideAwareConfig | None:
    path = Path(artifacts_dir) / _ARTIFACT
    if not path.exists():
        return None
    return SideAwareConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _fuse_score(
    probs: LayerProbabilities,
    *,
    ablation_mode: str | None,
    layer_mask: dict[str, bool],
    w_tab: float,
    w_seq: float,
    w_rl: float,
) -> float:
    tab_dir = 2.0 * probs.tabular_p_up - 1.0
    seq_dir = probs.sequence_trend_sign * (2.0 * probs.sequence_p_continuation - 1.0)
    rl_dir = float(np.clip(probs.rl_target_exposure, -1.0, 1.0))

    if not layer_mask.get("tabular", True):
        tab_dir = 0.0
    if not layer_mask.get("sequence", True):
        seq_dir = 0.0
    if not layer_mask.get("rl", True):
        rl_dir = 0.0

    if ablation_mode == "tabular_only":
        w_seq, w_rl = 0.0, 0.0
    elif ablation_mode == "tabular_sequence":
        w_rl = 0.0
    elif ablation_mode == "tabular_rl":
        w_seq = 0.0

    wsum = max(w_tab + w_seq + w_rl, 1e-9)
    fused = (w_tab * tab_dir + w_seq * seq_dir + w_rl * rl_dir) / wsum
    return float(np.clip(fused, -1.0, 1.0))


def _raw_confidence_for_source(
    source: str,
    side: str,
    *,
    fused: float,
    meta_pred: MetaPrediction | TwoStagePrediction | None,
    min_conviction: float,
    trade_threshold: float,
    direction_margin: float,
    direction_threshold: float,
) -> tuple[float, str | None]:
    if source in ("tabular_only", "no_meta"):
        if side == "LONG":
            if fused <= 0:
                return 0.0, "negative_fused_score"
            return float(np.clip((fused + 1.0) / 2.0, 0.0, 1.0)), None
        if fused >= 0:
            return 0.0, "positive_fused_score"
        return float(np.clip((-fused + 1.0) / 2.0, 0.0, 1.0)), None

    if source == "two_stage_meta" and isinstance(meta_pred, TwoStagePrediction):
        if side == "LONG":
            if meta_pred.p_trade < trade_threshold:
                return meta_pred.p_trade * meta_pred.p_long_given_trade, "below_trade_threshold"
            dc, reason = resolve_two_stage_direction(
                meta_pred.p_long_given_trade, meta_pred.p_short_given_trade, margin=direction_margin
            )
            if dc != CLASS_LONG:
                return meta_pred.p_trade * meta_pred.p_long_given_trade, reason or "not_long_direction"
            return meta_pred.p_trade * meta_pred.p_long_given_trade, None
        if meta_pred.p_trade < trade_threshold:
            return meta_pred.p_trade * meta_pred.p_short_given_trade, "below_trade_threshold"
        dc, reason = resolve_two_stage_direction(
            meta_pred.p_long_given_trade, meta_pred.p_short_given_trade, margin=direction_margin
        )
        if dc != CLASS_SHORT:
            return meta_pred.p_trade * meta_pred.p_short_given_trade, reason or "not_short_direction"
        return meta_pred.p_trade * meta_pred.p_short_given_trade, None

    if source == "legacy_3class_meta" and isinstance(meta_pred, MetaPrediction):
        if side == "LONG":
            dc, reason = resolve_directional_class(
                meta_pred.p_short, meta_pred.p_long, threshold=direction_threshold, margin=direction_margin
            )
            if dc != CLASS_LONG:
                return meta_pred.p_long, reason
            return meta_pred.p_long, None
        dc, reason = resolve_directional_class(
            meta_pred.p_short, meta_pred.p_long, threshold=direction_threshold, margin=direction_margin
        )
        if dc != CLASS_SHORT:
            return meta_pred.p_short, reason
        return meta_pred.p_short, None

    # Fallback heuristic
    conv = abs(fused)
    if conv < min_conviction:
        return conv, "below_min_conviction"
    return conv, None


def _ablation_for_source(source: str) -> str | None:
    if source == "tabular_only":
        return "tabular_only"
    if source == "no_meta":
        return None
    return None


def score_source_on_validation(
    source: str,
    *,
    F_val: np.ndarray,
    y_val: np.ndarray,
    meta_model: object | None,
    settings: Any,
    layer_mask: dict[str, bool],
    tab_p_val: np.ndarray | None = None,
    seq_cont_val: np.ndarray | None = None,
    seq_trend_val: np.ndarray | None = None,
    rl_val: np.ndarray | None = None,
    regime_val: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a decision source for LONG and SHORT separately on validation rows."""
    long_scores: list[float] = []
    short_scores: list[float] = []
    long_hits = long_n = short_hits = short_n = 0

    cfg = settings.fusion
    for i in range(len(F_val)):
        y = int(y_val[i])
        if tab_p_val is not None:
            p_up = float(tab_p_val[i])
            p_cont = float(seq_cont_val[i]) if seq_cont_val is not None else 0.5
            trend = float(seq_trend_val[i]) if seq_trend_val is not None else 0.0
            rl = float(rl_val[i]) if rl_val is not None else 0.0
            reg = regime_val[i] if regime_val is not None else np.array([0.33, 0.34, 0.33])
        else:
            vec = F_val[i]
            p_up, p_cont, trend, rl = float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])
            reg = vec[4:7]

        probs = LayerProbabilities(
            tabular_p_up=p_up,
            sequence_p_continuation=p_cont,
            sequence_trend_sign=trend,
            rl_target_exposure=rl,
        )
        fused = _fuse_score(
            probs,
            ablation_mode=_ablation_for_source(source),
            layer_mask=layer_mask,
            w_tab=cfg.w_tabular,
            w_seq=cfg.w_sequence,
            w_rl=cfg.w_rl,
        )

        meta_pred = None
        if source in ("two_stage_meta", "legacy_3class_meta") and meta_model is not None:
            mf = build_meta_features(p_up, p_cont, trend, rl, reg, layer_mask=layer_mask)
            if isinstance(meta_model, TwoStageMetaModel):
                meta_pred = meta_model.predict(
                    mf,
                    trade_threshold=cfg.meta_trade_threshold,
                    direction_margin=cfg.meta_direction_margin,
                )
            elif isinstance(meta_model, MetaModel):
                meta_pred = meta_model.predict(mf)

        raw_l, _ = _raw_confidence_for_source(
            source, "LONG", fused=fused, meta_pred=meta_pred,
            min_conviction=cfg.min_conviction, trade_threshold=cfg.meta_trade_threshold,
            direction_margin=cfg.meta_direction_margin, direction_threshold=cfg.meta_direction_threshold,
        )
        raw_s, _ = _raw_confidence_for_source(
            source, "SHORT", fused=fused, meta_pred=meta_pred,
            min_conviction=cfg.min_conviction, trade_threshold=cfg.meta_trade_threshold,
            direction_margin=cfg.meta_direction_margin, direction_threshold=cfg.meta_direction_threshold,
        )
        long_scores.append(raw_l)
        short_scores.append(raw_s)
        if raw_l > 0.05:
            long_n += 1
            if y == CLASS_LONG:
                long_hits += 1
        if raw_s > 0.05:
            short_n += 1
            if y == CLASS_SHORT:
                short_hits += 1

    long_arr = np.asarray(long_scores, dtype=float)
    short_arr = np.asarray(short_scores, dtype=float)
    return {
        "LONG_precision": float(long_hits / max(long_n, 1)),
        "SHORT_precision": float(short_hits / max(short_n, 1)),
        "LONG_mean_raw": float(long_arr.mean()) if long_arr.size else 0.0,
        "SHORT_mean_raw": float(short_arr.mean()) if short_arr.size else 0.0,
        "LONG_max_raw": float(long_arr.max()) if long_arr.size else 0.0,
        "SHORT_max_raw": float(short_arr.max()) if short_arr.size else 0.0,
        "LONG_publishable_proxy_ge_70": float((long_arr >= 0.70).mean()) if long_arr.size else 0.0,
        "SHORT_publishable_proxy_ge_70": float((short_arr >= 0.70).mean()) if short_arr.size else 0.0,
        "LONG_n": float(long_n),
        "SHORT_n": float(short_n),
    }


def select_sources_on_validation(source_scores: dict[str, dict[str, float]]) -> tuple[str, str, dict, dict]:
    """Pick best validation source per side without test leakage."""

    def _rank(side: str) -> str:
        ranked: list[tuple[float, float, float, str]] = []
        for source, metrics in source_scores.items():
            prec = metrics.get(f"{side}_precision", 0.0)
            pub = metrics.get(f"{side}_publishable_proxy_ge_70", 0.0)
            mean_raw = metrics.get(f"{side}_mean_raw", 0.0)
            max_raw = metrics.get(f"{side}_max_raw", mean_raw)
            score = 0.25 * prec + 0.55 * pub + 0.20 * mean_raw
            ranked.append((score, pub, max_raw, source))
        with_pub = [row for row in ranked if row[1] > 0]
        if with_pub:
            return max(with_pub, key=lambda x: x[0])[3]
        return max(ranked, key=lambda x: (x[2], x[0]))[3]

    long_src = _rank("LONG")
    short_src = _rank("SHORT")
    return long_src, short_src, source_scores.get(long_src, {}), source_scores.get(short_src, {})
