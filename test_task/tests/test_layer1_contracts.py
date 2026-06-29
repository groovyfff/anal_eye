"""Layer 1 - unit / property / contract tests locking the Layer 0 invariants.

These cover the cross-codebase seams reconciled in Layer 0:

1. Multi-asset null fallbacks in feature engineering (traditional assets send
   ``null`` for the derivative microstructure fields).
2. Sequence-layer window padding/truncation (30 / 48 / 64 candles) + the
   short-window warning emitted by the inference engine below the 48-candle floor.
3. The candidate contract round-trip from the backend producer's wire shape
   (``historical_ohlcv`` + ``timestamp``, no ``candles``/``ts``).

No GPU / trained artifacts required: the predictors degrade to neutral defaults
when no weights are present, so the plumbing is exercised deterministically.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from ae_brain.config import Settings
from ae_brain.contracts import FinalSignal, TradeCandidate
from ae_brain.features.engineering import FeatureEngineer, _coerce_series
from ae_brain.features.schema import FEATURE_NAMES
from ae_brain.inference import engine as engine_mod
from ae_brain.inference.engine import InferenceEngine, _engineer_latest
from ae_brain.layers.sequence import SEQ_CHANNELS, SequencePredictor
from ae_brain.training.synthetic import generate_synthetic_candles

# Derivative-only microstructure columns that arrive as JSON ``null`` for
# traditional assets (stock / forex / metal) coming from Yahoo Finance.
NULL_DERIVATIVE_COLS = (
    "funding_rate",
    "open_interest",
    "taker_buy_volume",
    "long_liq_notional",
    "short_liq_notional",
    "basis",
)

TRADITIONAL_ASSETS = ("stock", "forex", "metal")


def _null_derivative_candles(n: int, seed: int) -> pd.DataFrame:
    """Synthetic OHLCV with every derivative-only column nulled out."""
    candles = generate_synthetic_candles(n=n, seed=seed)
    for col in NULL_DERIVATIVE_COLS:
        candles[col] = None
    return candles


# --------------------------------------------------------------------------- #
# 1) Multi-asset null fallbacks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("asset_class", TRADITIONAL_ASSETS)
def test_compute_frame_null_microstructure_is_finite_and_neutral(asset_class: str) -> None:
    frame = FeatureEngineer().compute_frame(_null_derivative_candles(n=300, seed=11))

    # Full canonical contract, no NaN/inf leaks from the null inputs.
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert np.isfinite(frame.to_numpy()).all()

    # Funding (a derivative-only field) collapses to the neutral 0.0 default.
    assert np.allclose(frame["funding_rate"].to_numpy(), 0.0)
    assert np.allclose(frame["funding_z"].to_numpy(), 0.0)

    # Taker-buy volume defaults to 0.5 * volume -> a perfectly balanced book:
    # taker_buy_ratio == 0.5 and aggressor_imbalance == 0.0 on every row.
    assert np.allclose(frame["taker_buy_ratio"].to_numpy(), 0.5, atol=1e-6)
    assert np.allclose(frame["aggressor_imbalance"].to_numpy(), 0.0, atol=1e-6)


def test_coerce_series_taker_buy_defaults_to_half_volume() -> None:
    volume = np.array([10.0, 20.0, 0.0, 40.0])

    # Column entirely absent.
    s_absent = _coerce_series(pd.DataFrame({"volume": volume}), "taker_buy_volume", volume * 0.5)
    assert np.allclose(s_absent.to_numpy(), volume * 0.5)

    # Column present but full of JSON null (-> None -> NaN -> default).
    df_null = pd.DataFrame({"volume": volume, "taker_buy_volume": [None] * 4})
    s_null = _coerce_series(df_null, "taker_buy_volume", volume * 0.5)
    assert np.allclose(s_null.to_numpy(), volume * 0.5)

    # Scalar neutral default (funding-style).
    s_scalar = _coerce_series(pd.DataFrame({"funding_rate": [None, None, None]}), "funding_rate", 0.0)
    assert np.allclose(s_scalar.to_numpy(), 0.0)


@pytest.mark.parametrize("asset_class", TRADITIONAL_ASSETS)
def test_engineer_latest_forces_funding_zero_for_non_crypto(asset_class: str) -> None:
    candles = _null_derivative_candles(n=64, seed=7)
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")

    ctx = _engineer_latest(rows, z_window=50, asset_class=asset_class)

    assert ctx["funding_rate"] == 0.0  # forced 0.0 for non-derivative assets
    assert ctx["entry_price"] > 0.0
    assert np.isfinite(ctx["features"]).all()
    assert len(ctx["features"]) == len(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# 2) Sequence-window padding / truncation + short-window warning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_candles", [30, 48, 64])
def test_sequence_window_shape_is_stable(n_candles: int) -> None:
    s = Settings()
    predictor = SequencePredictor(s.model, s.gpu)
    assert predictor.window == 48  # the >=48 sequence-window invariant

    candles = generate_synthetic_candles(n=n_candles, seed=n_candles)

    # Short windows are left-padded, long windows truncated -> always (48, C).
    mat = predictor._to_window_array(candles)
    assert mat.shape == (predictor.window, len(SEQ_CHANNELS))

    # predict() is stable + finite even with no trained weights (neutral default).
    pred = predictor.predict(candles)
    assert np.isfinite(pred.p_continuation) and 0.0 <= pred.p_continuation <= 1.0
    assert np.isfinite(pred.trend_sign) and -1.0 <= pred.trend_sign <= 1.0


class _LogRecorder:
    """Minimal structlog-compatible stub that records emitted events."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, event: str, **_: object) -> None:
        self.warnings.append(event)

    # Everything else is a no-op for the test.
    def info(self, *_: object, **__: object) -> None: ...
    def error(self, *_: object, **__: object) -> None: ...
    def exception(self, *_: object, **__: object) -> None: ...


def _crypto_candidate(n_candles: int) -> TradeCandidate:
    candles = generate_synthetic_candles(n=n_candles, seed=100 + n_candles)
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")
    return TradeCandidate(
        symbol="BTCUSDT", interval="5m", candles=rows, signal_log_db_id=0, asset_class="crypto"
    )


def test_engine_warns_only_below_sequence_window(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _LogRecorder()
    monkeypatch.setattr(engine_mod, "log", recorder)

    settings = Settings()
    settings.executor.process_workers = 0  # keep it in-process (thread pool only)
    eng = InferenceEngine(settings, db=None)
    eng.load_models()

    async def _evaluate(candidate: TradeCandidate) -> FinalSignal:
        return await eng.evaluate(candidate)

    try:
        sig_short = asyncio.run(_evaluate(_crypto_candidate(30)))
        assert isinstance(sig_short, FinalSignal)  # warns, never rejects
        assert "engine.candles.short" in recorder.warnings

        recorder.warnings.clear()
        sig_ok = asyncio.run(_evaluate(_crypto_candidate(64)))
        assert isinstance(sig_ok, FinalSignal)
        assert "engine.candles.short" not in recorder.warnings
    finally:
        asyncio.run(eng.shutdown())


# --------------------------------------------------------------------------- #
# 3) Candidate contract round-trip (backend producer wire shape)
# --------------------------------------------------------------------------- #
def test_from_message_accepts_backend_historical_ohlcv_payload() -> None:
    candles = generate_synthetic_candles(n=64, seed=3)
    # Backend producer shape: 'historical_ohlcv' with an ISO 'timestamp' per row,
    # NO 'candles' key and NO 'ts' key.
    historical_ohlcv = [
        {
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": float(o),
            "high": float(h),
            "low": float(low),
            "close": float(c),
            "volume": float(v),
        }
        for ts, o, h, low, c, v in zip(
            candles["ts"],
            candles["open"],
            candles["high"],
            candles["low"],
            candles["close"],
            candles["volume"],
        )
    ]
    payload = {
        "symbol": "GC=F",
        "asset_class": "metal",
        "signal_id": "f1d2e3c4-0000-4000-8000-000000000001",
        "signal_log_db_id": 777,
        "historical_ohlcv": historical_ohlcv,
        # deliberately omit "candles" and "ts" to mirror the real producer
    }

    candidate = TradeCandidate.from_message(payload)  # must NOT raise KeyError

    assert len(candidate.candles) == 64
    # 'timestamp' must be mapped onto 'ts' so the session time features work.
    assert candidate.candles[0]["ts"] == historical_ohlcv[0]["timestamp"]
    assert candidate.signal_id == "f1d2e3c4-0000-4000-8000-000000000001"
    assert candidate.signal_log_db_id == 777
    assert candidate.asset_class == "metal"
    assert candidate.is_derivative is False  # metal carries no funding/OI/CVD

    # The mapped window is directly consumable by the feature engineer.
    frame = FeatureEngineer().compute_frame(pd.DataFrame(candidate.candles))
    assert np.isfinite(frame.to_numpy()).all()


def test_from_message_prefers_candles_over_historical_ohlcv() -> None:
    candles = generate_synthetic_candles(n=50, seed=4)
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")
    payload = {
        "symbol": "BTCUSDT",
        "candles": rows,
        "historical_ohlcv": [{"timestamp": "2026-01-01T00:00:00Z", "close": 1.0}],
        "signal_log_db_id": 5,
    }
    candidate = TradeCandidate.from_message(payload)
    assert len(candidate.candles) == 50  # 'candles' wins when both are present
