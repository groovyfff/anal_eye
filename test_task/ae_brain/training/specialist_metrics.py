"""Production-focused metrics for side specialists: top-k precision, EV, calibration ceiling."""

from __future__ import annotations

from typing import Any

import numpy as np


def _dist_stats(values: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "mean": float(np.mean(v)),
        "p50": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(np.max(v)),
    }


def top_k_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(n * frac)))
    thr = np.partition(scores, -k)[-k]
    return scores >= thr


def precision_at_top_frac(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    mask = top_k_mask(scores, frac)
    if not mask.any():
        return 0.0
    return float(np.mean(y_true[mask]))


def ev_at_top_frac(ev: np.ndarray, scores: np.ndarray, frac: float) -> float:
    mask = top_k_mask(scores, frac)
    if not mask.any():
        return 0.0
    return float(np.sum(ev[mask]))


def calibration_bucket_report(
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    *,
    n_buckets: int = 10,
) -> list[dict[str, Any]]:
    cal = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
    y = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (cal >= lo) & (cal < hi if i < n_buckets - 1 else cal <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"[{lo:.2f},{hi:.2f})",
                "n": int(mask.sum()),
                "realized_precision": float(y[mask].mean()),
                "mean_calibrated": float(cal[mask].mean()),
                "mean_raw": float(raw[mask].mean()),
            }
        )
    return rows


def confidence_ceiling_report(
    *,
    side: str,
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    ev_usd: np.ndarray | None = None,
    symbols: np.ndarray | None = None,
    regime_ids: np.ndarray | None = None,
    publish_threshold: float = 0.70,
) -> dict[str, Any]:
    """Full ceiling diagnostics for one side on a validation slice."""
    y = np.asarray(y_true, dtype=int)
    raw = np.asarray(raw, dtype=float)
    cal = np.asarray(calibrated, dtype=float)
    ev = np.asarray(ev_usd, dtype=float) if ev_usd is not None else np.zeros_like(raw)

    top_fracs = (0.01, 0.02, 0.05, 0.10)
    raw_top: dict[str, float] = {}
    cal_top: dict[str, float] = {}
    raw_ev_top: dict[str, float] = {}
    cal_ev_top: dict[str, float] = {}
    for f in top_fracs:
        key = f"top_{int(f * 100)}pct"
        raw_top[key] = precision_at_top_frac(y, raw, f)
        cal_top[key] = precision_at_top_frac(y, cal, f)
        raw_ev_top[key] = ev_at_top_frac(ev, raw, f)
        cal_ev_top[key] = ev_at_top_frac(ev, cal, f)

    buckets = calibration_bucket_report(y, raw, cal)
    max_bucket_prec = max((b["realized_precision"] for b in buckets), default=0.0)
    ge_70_buckets = [b for b in buckets if b["mean_calibrated"] >= publish_threshold - 1e-9]

    out: dict[str, Any] = {
        "side": side,
        "raw_score": _dist_stats(raw),
        "calibrated_confidence": _dist_stats(cal),
        "profitable_rate_overall": float(y.mean()) if y.size else 0.0,
        "raw_precision_at_top_k": raw_top,
        "calibrated_precision_at_top_k": cal_top,
        "raw_ev_at_top_k": raw_ev_top,
        "calibrated_ev_at_top_k": cal_ev_top,
        "calibration_buckets": buckets,
        "max_bucket_realized_precision": max_bucket_prec,
        "buckets_with_calibrated_ge_70": ge_70_buckets,
        "any_bucket_precision_ge_70": max_bucket_prec >= publish_threshold,
        "calibrated_max_ge_70": bool(cal.size and float(np.max(cal)) >= publish_threshold),
    }

    if symbols is not None:
        sym = np.asarray(symbols)
        per_sym: dict[str, Any] = {}
        for s in sorted(set(sym)):
            m = sym == s
            per_sym[str(s)] = {
                "raw": _dist_stats(raw[m]),
                "calibrated": _dist_stats(cal[m]),
                "top_5pct_precision_raw": precision_at_top_frac(y[m], raw[m], 0.05),
            }
        out["per_symbol"] = per_sym

    if regime_ids is not None:
        reg = np.asarray(regime_ids)
        per_reg: dict[str, Any] = {}
        for r in sorted(set(reg)):
            m = reg == r
            per_reg[str(r)] = {
                "raw": _dist_stats(raw[m]),
                "calibrated": _dist_stats(cal[m]),
                "top_5pct_precision_raw": precision_at_top_frac(y[m], raw[m], 0.05),
            }
        out["per_regime"] = per_reg

    if out["any_bucket_precision_ge_70"] and not out["calibrated_max_ge_70"]:
        out["ceiling_diagnosis"] = (
            "validation_buckets_show_precision_ge_70_but_isotonic_max_below_70"
        )
    elif not out["any_bucket_precision_ge_70"]:
        out["ceiling_diagnosis"] = "model_lacks_70pct_precision_even_in_top_buckets"
    else:
        out["ceiling_diagnosis"] = "calibration_can_reach_70"

    return out


def simulate_publishable_ev(
    long_cal: np.ndarray,
    short_cal: np.ndarray,
    long_ev: np.ndarray,
    short_ev: np.ndarray,
    *,
    threshold: float = 0.70,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    """Simulate fusion publishable counts/EV on a validation matrix (no test leakage)."""
    n = len(long_cal)
    pub_long = pub_short = 0
    ev_long = ev_short = 0.0
    internal_long = internal_short = skip = 0

    for i in range(n):
        lv = long_cal[i] >= threshold and long_ev[i] > 0
        sv = short_cal[i] >= threshold and short_ev[i] > 0
        if lv and not sv:
            internal_long += 1
            pub_long += 1
            ev_long += long_ev[i]
        elif sv and not lv:
            internal_short += 1
            pub_short += 1
            ev_short += short_ev[i]
        elif lv and sv:
            u_l = long_cal[i] * max(long_ev[i], 0.0)
            u_s = short_cal[i] * max(short_ev[i], 0.0)
            if abs(u_l - u_s) < ambiguity_margin * max(u_l, u_s, 1e-9):
                skip += 1
            elif u_l > u_s:
                internal_long += 1
                pub_long += 1
                ev_long += long_ev[i]
            else:
                internal_short += 1
                pub_short += 1
                ev_short += short_ev[i]
        else:
            skip += 1

    return {
        "internal_LONG": internal_long,
        "internal_SHORT": internal_short,
        "SKIP": skip,
        f"publishable_LONG_ge_{threshold:.2f}": pub_long,
        f"publishable_SHORT_ge_{threshold:.2f}": pub_short,
        "publishable_EV_LONG": ev_long,
        "publishable_EV_SHORT": ev_short,
        "publishable_EV_total": ev_long + ev_short,
    }


def _publishable_counts_at_thresholds(
    calibrated: np.ndarray, thresholds: tuple[float, ...]
) -> dict[str, int]:
    cal = np.asarray(calibrated, dtype=float)
    out: dict[str, int] = {}
    for t in thresholds:
        out[f"ge_{t:.2f}"] = int((cal >= t).sum())
    return out


def calibration_ceiling_summary(
    long_cal: np.ndarray,
    short_cal: np.ndarray,
    *,
    publish_threshold: float = 0.70,
    diagnostics_thresholds: tuple[float, ...] = (0.65, 0.70, 0.75),
) -> dict[str, Any]:
    """Side-specific calibration ceiling diagnostics.

    Reports the maximum / p90 / p95 / p99 calibrated confidence per side, how many
    candidates cross the publish threshold per side, and a clear flag when the LONG
    calibrated ceiling is too low to reach the publish gate.
    """
    out: dict[str, Any] = {}
    for side, cal in (("LONG", long_cal), ("SHORT", short_cal)):
        cal = np.asarray(cal, dtype=float)
        if cal.size == 0:
            out[side] = {"n": 0, "max": 0.0, "crossing_publish_threshold": 0}
            continue
        stats = _dist_stats(cal)
        crossing = int((cal >= publish_threshold).sum())
        out[side] = {
            **stats,
            "crossing_publish_threshold": crossing,
            "publishable_counts_at_thresholds": _publishable_counts_at_thresholds(
                cal, diagnostics_thresholds
            ),
            "calibrated_max_ge_publish_threshold": bool(stats["max"] >= publish_threshold),
            "ceiling_too_low": bool(stats["max"] < publish_threshold),
        }
    long_max = out["LONG"].get("max", 0.0)
    out["long_ceiling_diagnosis"] = (
        "long_calibrated_max_below_publish_threshold"
        if long_max < publish_threshold
        else "long_calibrated_max_can_reach_publish_threshold"
    )
    return out


def second_pass_threshold_report(
    long_cal: np.ndarray,
    short_cal: np.ndarray,
    long_ev: np.ndarray,
    short_ev: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.65, 0.70, 0.75),
) -> dict[str, Any]:
    """Diagnostic-only report of publishable counts/EV/share at multiple thresholds.

    Promotion always uses the fixed 0.70 gate; this is for diagnosis only and must
    NOT be used to promote on a lower threshold.
    """
    per_threshold: dict[str, Any] = {}
    for t in thresholds:
        sim = simulate_publishable_ev(
            long_cal, short_cal, long_ev, short_ev, threshold=t, ambiguity_margin=0.05
        )
        pub_long = sim[f"publishable_LONG_ge_{t:.2f}"]
        pub_short = sim[f"publishable_SHORT_ge_{t:.2f}"]
        total = pub_long + pub_short
        per_threshold[f"{t:.2f}"] = {
            "threshold": t,
            "publishable_LONG": pub_long,
            "publishable_SHORT": pub_short,
            "publishable_total": total,
            "long_share": float(pub_long / total) if total else 0.0,
            "short_share": float(pub_short / total) if total else 0.0,
            "publishable_EV_LONG": sim["publishable_EV_LONG"],
            "publishable_EV_SHORT": sim["publishable_EV_SHORT"],
            "publishable_EV_total": sim["publishable_EV_total"],
        }
    return {
        "diagnostic_only": True,
        "promotion_threshold_fixed_at": 0.70,
        "note": "Thresholds 0.65 are diagnostic only; promotion never uses 0.65.",
        "per_threshold": per_threshold,
    }


def side_balance_report(
    *,
    y_long_train: np.ndarray,
    y_short_train: np.ndarray,
    y_long_val: np.ndarray,
    y_short_val: np.ndarray,
    y_long_test: np.ndarray | None = None,
    y_short_test: np.ndarray | None = None,
    long_cal: np.ndarray | None = None,
    short_cal: np.ndarray | None = None,
    long_ev: np.ndarray | None = None,
    short_ev: np.ndarray | None = None,
    symbols_train: np.ndarray | None = None,
    symbols_val: np.ndarray | None = None,
    publish_threshold: float = 0.70,
    diagnostics_thresholds: tuple[float, ...] = (0.70, 0.75, 0.80),
) -> dict[str, Any]:
    """Full LONG/SHORT side-balance diagnostics across train/val(/test) splits.

    Reports:
    * LONG/SHORT label counts and positive class ratio per split.
    * Publishable LONG/SHORT counts at several confidence thresholds.
    * LONG/SHORT EV separately.
    * Per-symbol publishable LONG/SHORT counts on validation.
    * Top symbols causing SHORT dominance.
    """
    def _counts(y_l: np.ndarray, y_s: np.ndarray) -> dict[str, Any]:
        yl = np.asarray(y_l, dtype=int)
        ys = np.asarray(y_s, dtype=int)
        n = max(yl.size, ys.size)
        skip = n - int(np.logical_or(yl, ys).sum()) if n else 0
        return {
            "LONG_profitable": int(yl.sum()),
            "SHORT_profitable": int(ys.sum()),
            "SKIP": skip,
            "n": int(n),
            "LONG_positive_ratio": float(yl.mean()) if yl.size else 0.0,
            "SHORT_positive_ratio": float(ys.mean()) if ys.size else 0.0,
        }

    report: dict[str, Any] = {
        "publish_threshold": publish_threshold,
        "label_counts": {
            "train": _counts(y_long_train, y_short_train),
            "validation": _counts(y_long_val, y_short_val),
        },
    }
    if y_long_test is not None and y_short_test is not None:
        report["label_counts"]["test"] = _counts(y_long_test, y_short_test)

    if long_cal is not None and short_cal is not None:
        report["publishable_counts"] = {
            "LONG": _publishable_counts_at_thresholds(
                np.asarray(long_cal, dtype=float), diagnostics_thresholds
            ),
            "SHORT": _publishable_counts_at_thresholds(
                np.asarray(short_cal, dtype=float), diagnostics_thresholds
            ),
        }
    if long_ev is not None and short_ev is not None:
        report["ev_separate"] = {
            "LONG": float(np.sum(np.asarray(long_ev, dtype=float))),
            "SHORT": float(np.sum(np.asarray(short_ev, dtype=float))),
        }

    # Per-symbol publishable counts + SHORT dominance ranking on validation.
    if (
        symbols_val is not None
        and long_cal is not None
        and short_cal is not None
    ):
        sym = np.asarray(symbols_val)
        lcal = np.asarray(long_cal, dtype=float)
        scal = np.asarray(short_cal, dtype=float)
        l_pub = lcal >= publish_threshold
        s_pub = scal >= publish_threshold
        per_sym: dict[str, dict[str, int]] = {}
        for s in sorted(set(sym)):
            m = sym == s
            per_sym[str(s)] = {
                "LONG": int(l_pub[m].sum()),
                "SHORT": int(s_pub[m].sum()),
            }
        report["per_symbol_publishable_validation"] = per_sym
        dominance = sorted(
            (
                {"symbol": s, "SHORT": v["SHORT"], "LONG": v["LONG"],
                 "short_minus_long": v["SHORT"] - v["LONG"]}
                for s, v in per_sym.items()
            ),
            key=lambda d: d["short_minus_long"],
            reverse=True,
        )
        report["top_symbols_short_dominance"] = dominance[:20]

    return report
