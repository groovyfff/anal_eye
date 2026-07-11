"""Async inference engine - orchestrates the 4 layers end-to-end.

Concurrency model
-----------------
The asyncio event loop stays responsive by offloading *all* heavy / blocking
work to executors:

* **ProcessPoolExecutor** - CPU-bound feature engineering (TA-Lib, numpy). True
  parallelism, GIL-immune.
* **ThreadPoolExecutor**   - model inference (LightGBM C++, torch/ONNX). These
  release the GIL during compute, so threads give real concurrency without the
  cost of shipping CUDA contexts / large arrays across processes.

``evaluate`` is the single public coroutine: candidate -> FinalSignal.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ae_brain.config import Settings
from ae_brain.contracts import FinalSignal, LayerProbabilities, TradeCandidate
from ae_brain.data.database import Database
from ae_brain.execution.timing import resolve_execution_timing
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.features.schema import FEATURE_NAMES, REGIME_ONEHOT_NAMES, n_features
from ae_brain.layers.fusion import FusionContext, FusionLayer
from ae_brain.layers.meta import MetaModel, TwoStageMetaModel, load_meta_model
from ae_brain.layers.side_aware import load_side_aware_config
from ae_brain.layers.side_specialists import load_side_specialists
from ae_brain.training.calibration import ConfidenceCalibrator, SideCalibrators
from ae_brain.training.regime_filter import TrainingRegimeConfig, load_training_regime
from ae_brain.layers.risk_agent import RiskAgent
from ae_brain.layers.sequence import SequencePredictor
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.risk.costs import CostModel
from ae_brain.risk.ev_gate import EVGate
from ae_brain.risk.sizing import PositionSizer
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.engine")


# --- module-level worker fn so it is picklable for ProcessPoolExecutor ------
def _last_float(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Return the last value of a column as a float, null/missing-safe.

    Handles traditional assets where derivatives fields (funding_rate, ...)
    arrive as JSON ``null`` -> pandas ``None``/``NaN`` -> would otherwise raise
    ``TypeError`` on ``float(None)``.
    """
    if col not in df.columns:
        return default
    val = pd.to_numeric(df[col], errors="coerce").iloc[-1]
    return float(val) if pd.notna(val) else default


# Per-process cache for the (read-only) regime model so the ProcessPoolExecutor
# workers each load it at most once.
_REGIME_CACHE: dict[str, object] = {}


def _get_regime_model(artifacts_dir: str | None):
    if not artifacts_dir:
        return None
    key = str(artifacts_dir)
    if key not in _REGIME_CACHE:
        from ae_brain.features.regime import RegimeModel

        _REGIME_CACHE[key] = RegimeModel.try_load(Path(artifacts_dir))
    return _REGIME_CACHE[key]


def _engineer_latest(
    candle_rows: list[dict],
    z_window: int,
    asset_class: str = "crypto",
    artifacts_dir: str | None = None,
    regime_enabled: bool = True,
) -> dict:
    """Compute the latest feature vector + entry/atr/regime context (process-safe).

    Robust to ``null`` microstructure fields (non-crypto assets): all extraction
    goes through ``_last_float`` and the FeatureEngineer maps nulls to neutral
    defaults internally. For non-derivative assets funding is forced to 0.0.

    When a fitted regime model is available it is attached to the FeatureEngineer
    so the canonical vector carries the regime one-hot; the per-row regime is
    also returned so the sequence model receives regime channels.
    """
    regime_model = _get_regime_model(artifacts_dir) if regime_enabled else None
    eng = FeatureEngineer(z_window=z_window, regime_model=regime_model)
    df = pd.DataFrame(candle_rows)
    if "open_time" in df and "ts" not in df:
        df = df.rename(columns={"open_time": "ts"})
    frame = eng.compute_frame(df)
    latest = frame.iloc[-1]

    signal_reference_price = _last_float(df, "close", 0.0)
    atr = float(latest.get("atr_14", 0.0)) or (signal_reference_price * 0.005)
    is_derivative = str(asset_class).lower() == "crypto"
    funding_rate = _last_float(df, "funding_rate", 0.0) if is_derivative else 0.0

    regime_rows = frame[list(REGIME_ONEHOT_NAMES)].to_numpy(dtype=float)
    regime_onehot = regime_rows[-1].tolist() if len(regime_rows) else [0.0, 0.0, 0.0]

    from ae_brain.training.specialist_features import augment_symbol_frame

    prices = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = frame["atr_14"].to_numpy(dtype=float)
    feat_arr = frame.to_numpy(dtype=float)
    extra_arrays = augment_symbol_frame(feat_arr, prices, atr_arr, high, low)
    specialist_extra = {k: float(v[-1]) for k, v in extra_arrays.items()}
    if "mark_close" in df.columns and "index_close" in df.columns:
        mark = pd.to_numeric(df["mark_close"], errors="coerce").iloc[-1]
        index = pd.to_numeric(df["index_close"], errors="coerce").iloc[-1]
        if index and not pd.isna(index):
            specialist_extra["basis_pct"] = float((mark - index) / index) if not pd.isna(mark) else 0.0

    btc_specialist_ctx: dict[str, float] = {}
    sym = str(df.get("symbol", pd.Series([""])).iloc[-1]) if "symbol" in df.columns else ""
    if sym == "BTCUSDT":
        btc_specialist_ctx = {
            "btc_ret_15": float(latest.get("ret_15", 0.0)),
            "btc_vol_z": float(latest.get("vol_z", 0.0)),
            "btc_regime_trend": float(regime_onehot[0]) if regime_onehot else 0.0,
        }

    return {
        "features": latest.to_numpy(dtype=np.float32),
        "signal_reference_price": signal_reference_price,
        "entry_price": signal_reference_price,
        "atr": atr,
        "funding_rate": funding_rate,
        "vol_z": float(latest.get("vol_z", 0.0)),
        "regime_onehot": regime_onehot,
        "regime_rows": regime_rows.tolist(),
        "specialist_extra": specialist_extra,
        "btc_specialist_ctx": btc_specialist_ctx,
    }


class InferenceEngine:
    def __init__(self, settings: Settings, db: Optional[Database] = None) -> None:
        self._s = settings
        self._db = db

        cost_model = CostModel(settings.cost)
        self._ev_gate = EVGate(cost_model, min_ev_usd=settings.fusion.min_ev_usd)
        self._sizer = PositionSizer(settings.risk)
        # Meta-model is the directional authority when trained/loaded; the
        # fusion layer falls back to the heuristic EV gate otherwise.
        self._meta = load_meta_model(settings.model.artifacts_dir, prefer_two_stage=settings.model.meta_prefer_two_stage) if settings.model.use_meta_model else None
        self._meta_legacy = MetaModel().load(settings.model.artifacts_dir)
        self._meta_two_stage = TwoStageMetaModel().load(settings.model.artifacts_dir)
        if not self._meta_legacy.is_ready():
            self._meta_legacy = None
        if not self._meta_two_stage.is_ready():
            self._meta_two_stage = None
        self._calibrator = ConfidenceCalibrator(settings.model.calibration_method)
        self._side_calibrators = SideCalibrators(settings.model.calibration_method)
        self._side_aware_config = load_side_aware_config(settings.model.artifacts_dir)
        self._side_specialists = load_side_specialists(settings.model.artifacts_dir)
        layer_mask = {"tabular": True, "sequence": True, "rl": True}
        mask_path = Path(settings.model.artifacts_dir) / "meta_layer_mask.json"
        if mask_path.exists():
            layer_mask = json.loads(mask_path.read_text(encoding="utf-8"))
        training_regime = load_training_regime(Path(settings.model.artifacts_dir))
        self._fusion = FusionLayer(
            settings.fusion,
            settings.risk,
            self._ev_gate,
            self._sizer,
            meta_model=self._meta,
            meta_legacy=self._meta_legacy,
            meta_two_stage=self._meta_two_stage,
            confidence_calibrator=self._calibrator,
            side_calibrators=self._side_calibrators,
            side_aware_config=self._side_aware_config,
            side_specialists=self._side_specialists,
            layer_mask=layer_mask,
            training_regime=training_regime,
        )

        self._tabular = TabularPredictor(settings.model)
        self._sequence = SequencePredictor(settings.model, settings.gpu)
        self._rl = RiskAgent(settings.model)

        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.executor.thread_workers, thread_name_prefix="aeb-infer"
        )
        self._process_pool: Optional[ProcessPoolExecutor] = None
        if settings.executor.process_workers > 0:
            self._process_pool = ProcessPoolExecutor(max_workers=settings.executor.process_workers)
        self._models_loaded = False

    # ------------------------------------------------------------------ #
    def load_models(self) -> None:
        artifacts = self._s.model.artifacts_dir
        self._tabular.load(artifacts)
        self._sequence.load(artifacts)
        self._rl.load(artifacts)
        if self._meta is not None:
            self._meta = load_meta_model(artifacts, prefer_two_stage=self._s.model.meta_prefer_two_stage)
        self._meta_legacy = MetaModel().load(artifacts)
        self._meta_two_stage = TwoStageMetaModel().load(artifacts)
        if not self._meta_legacy.is_ready():
            self._meta_legacy = None
        if not self._meta_two_stage.is_ready():
            self._meta_two_stage = None
        self._calibrator.load(artifacts)
        self._side_calibrators.load(artifacts)
        self._side_aware_config = load_side_aware_config(artifacts)
        self._side_specialists = load_side_specialists(artifacts)
        mask_path = Path(artifacts) / "meta_layer_mask.json"
        if mask_path.exists():
            self._fusion._layer_mask = json.loads(mask_path.read_text(encoding="utf-8"))
        self._fusion._meta = self._meta
        self._fusion._meta_legacy = self._meta_legacy
        self._fusion._meta_two_stage = self._meta_two_stage
        self._fusion._calibrator = self._calibrator
        self._fusion._side_calibrators = self._side_calibrators
        self._fusion._side_aware_config = self._side_aware_config
        self._fusion._side_specialists = self._side_specialists
        self._fusion._training_regime = load_training_regime(Path(artifacts))
        # Warm the per-process regime cache
        regime = _get_regime_model(str(artifacts)) if self._s.model.regime_enabled else None
        log.info(
            "engine.models.loaded",
            tabular=self._tabular.is_ready(),
            sequence=self._sequence.is_ready(),
            rl=self._rl.is_ready(),
            regime=bool(regime is not None and getattr(regime, "is_ready", lambda: False)()),
            meta=bool(self._meta is not None and self._meta.is_ready()),
        )
        self._models_loaded = True

    def is_ready(self) -> bool:
        """True after :meth:`load_models` has been invoked (inference may use fallbacks)."""
        return self._models_loaded

    async def shutdown(self) -> None:
        self._thread_pool.shutdown(wait=True)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    async def evaluate(self, candidate: TradeCandidate) -> FinalSignal:
        loop = asyncio.get_running_loop()

        # Candle-window check: the sequence layer needs >= sequence_window candles
        # (>=48 by default). We warn rather than reject; the SequencePredictor
        # left-pads short windows so a single thin message still produces a signal.
        required = self._s.model.sequence_window
        if len(candidate.candles) < required:
            log.warning(
                "engine.candles.short",
                symbol=candidate.symbol,
                got=len(candidate.candles),
                required=required,
            )

        # 1) Feature engineering (CPU-bound -> process pool when available).
        pool = self._process_pool or self._thread_pool
        ctx_data = await loop.run_in_executor(
            pool,
            _engineer_latest,
            candidate.candles,
            100,
            candidate.asset_class,
            str(self._s.model.artifacts_dir),
            self._s.model.regime_enabled,
        )

        features: np.ndarray = ctx_data["features"]
        candle_df = pd.DataFrame(candidate.candles)
        # Attach per-bar regime channels so the sequence model receives them.
        reg_rows = ctx_data.get("regime_rows")
        if reg_rows is not None and len(reg_rows) == len(candle_df):
            reg_arr = np.asarray(reg_rows, dtype=float)
            for j, name in enumerate(REGIME_ONEHOT_NAMES):
                candle_df[name] = reg_arr[:, j]

        # 2) Layer inference concurrently on the thread pool.
        tab_fut = loop.run_in_executor(self._thread_pool, self._tabular.predict, features)
        seq_fut = loop.run_in_executor(self._thread_pool, self._sequence.predict, candle_df)
        rl_obs = self._build_rl_obs(features, ctx_data, candidate)
        rl_fut = loop.run_in_executor(self._thread_pool, self._rl.predict, rl_obs)

        tab_pred, seq_pred, rl_pred = await asyncio.gather(tab_fut, seq_fut, rl_fut)

        probs = LayerProbabilities(
            tabular_p_up=tab_pred.p_up,
            sequence_p_continuation=seq_pred.p_continuation,
            sequence_trend_sign=seq_pred.trend_sign,
            rl_target_exposure=rl_pred.target_exposure,
            rl_state_value=rl_pred.state_value,
        )

        # 3) Correlation context (portfolio risk constraint).
        correlated_exposure = await self._correlated_exposure(candidate.symbol)

        timing = resolve_execution_timing(
            candidate.candles,
            candidate.meta,
            slippage_bps=self._s.cost.base_slippage_bps,
        )

        fusion_ctx = FusionContext(
            symbol=candidate.symbol,
            entry_price=timing.execution_price,
            atr=ctx_data["atr"],
            funding_rate_8h=ctx_data["funding_rate"],
            adv_usd=candidate.meta.get("adv_usd"),
            holding_hours=float(candidate.meta.get("expected_holding_hours", 8.0)),
            correlated_exposure=correlated_exposure,
            correlation_id=candidate.correlation_id,
            regime_onehot=tuple(ctx_data.get("regime_onehot", (0.0, 0.0, 0.0))),
            tabular_features=features,
            specialist_extra=ctx_data.get("specialist_extra"),
            btc_specialist_ctx=candidate.meta.get("btc_specialist_ctx") or ctx_data.get("btc_specialist_ctx"),
            vol_z=ctx_data.get("vol_z"),
            signal_reference_price=timing.signal_reference_price,
            execution_price_source=timing.execution_price_source,
            signal_candle_open_time=timing.signal_candle_open_time,
            signal_candle_close_time=timing.signal_candle_close_time,
            execution_time=timing.execution_time,
        )

        # 4) Fuse + EV gate -> deterministic decision.
        signal = self._fusion.decide(probs, fusion_ctx)

        # Propagate candidate identity/routing so the published signal.final
        # satisfies the tracker-service contract (tp/sl come from FinalSignal).
        signal.signal_id = candidate.signal_id or candidate.correlation_id
        signal.asset_class = candidate.asset_class
        signal.signal_log_db_id = candidate.signal_log_db_id
        signal.entry_reference = timing.execution_price
        signal.execution_price = timing.execution_price
        signal.signal_reference_price = timing.signal_reference_price
        signal.signal_candle_open_time = timing.signal_candle_open_time
        signal.signal_candle_close_time = timing.signal_candle_close_time
        signal.execution_time = timing.execution_time
        signal.execution_price_source = timing.execution_price_source
        signal.components = dict(signal.components or {})
        signal.components["execution_timing"] = timing.to_dict()

        # 5) Best-effort audit log (never block the decision on logging failure).
        await self._log_signal(candidate, features, probs, signal)
        return signal

    # ------------------------------------------------------------------ #
    def _build_rl_obs(self, features: np.ndarray, ctx_data: dict, candidate: TradeCandidate) -> np.ndarray:
        """Recreate the env observation: features + [pos, equity-1, atr_pct, corr]."""
        atr_pct = ctx_data["atr"] / ctx_data["signal_reference_price"] if ctx_data["signal_reference_price"] > 0 else 0.0
        position = float(candidate.meta.get("current_position", 0.0))
        corr = float(candidate.meta.get("correlated_exposure", 0.0))
        extra = np.array([position, 0.0, atr_pct, corr], dtype=np.float32)
        return np.concatenate([features.astype(np.float32), extra])

    async def _correlated_exposure(self, symbol: str) -> float:
        if self._db is None:
            return 0.0
        try:
            corrs = await self._db.fetch_correlations(symbol)
        except Exception as exc:  # pragma: no cover - logging path
            log.warning("engine.corr.failed", err=str(exc))
            return 0.0
        thr = self._s.risk.correlation_threshold
        return float(sum(abs(c) for c in corrs.values() if abs(c) >= thr))

    async def _log_signal(self, candidate, features, probs, signal) -> None:
        """Persist ensemble outputs.

        PRD/ТЗ #2: when the backend supplied a pre-inserted row id, UPDATE that
        row. Otherwise (local/dev/API) fall back to an INSERT so the path still
        works end-to-end. Never blocks the decision on a logging failure.
        """
        if self._db is None:
            return
        try:
            feat_dict = {name: float(v) for name, v in zip(FEATURE_NAMES, features)}
            if candidate.signal_log_db_id and candidate.signal_log_db_id > 0:
                await self._db.update_signal_log(
                    signal_log_db_id=candidate.signal_log_db_id,
                    features=feat_dict,
                    layer_probs=probs.as_dict(),
                    signal=signal,
                    asset_class=candidate.asset_class,
                )
            else:
                await self._db.log_signal(
                    symbol=candidate.symbol,
                    interval=candidate.interval,
                    features=feat_dict,
                    layer_probs=probs.as_dict(),
                    signal=signal,
                    asset_class=candidate.asset_class,
                )
        except Exception as exc:  # pragma: no cover
            log.warning("engine.log.failed", err=str(exc))

    @staticmethod
    def expected_feature_dim() -> int:
        return n_features()
