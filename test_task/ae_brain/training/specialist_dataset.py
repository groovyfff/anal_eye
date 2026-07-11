"""Collect side-specialist training rows with EV-aware labels and extended features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ae_brain.contracts import Side
from ae_brain.layers.meta import build_meta_features
from ae_brain.training.labels import LabelConfig, _net_reward_at_barrier, side_specific_profitable_labels
from ae_brain.training.side_configs import SideLabelConfig, SideLabelConfigPair
from ae_brain.training.specialist_features import (
    SPECIALIST_FEATURE_NAMES,
    augment_symbol_frame,
    build_specialist_feature_row,
    liquidity_bucket,
)
from ae_brain.config import CostConfig
from ae_brain.risk.costs import CostModel


@dataclass
class SpecialistDataset:
    F: np.ndarray
    y_long: np.ndarray
    y_short: np.ndarray
    ev_long: np.ndarray
    ev_short: np.ndarray
    symbols: np.ndarray
    regime_ids: np.ndarray
    timestamps: np.ndarray
    label_config: dict[str, Any]

    def __len__(self) -> int:
        return len(self.F)

    def train_val_cuts(self, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple[int, int]:
        n = len(self.F)
        cut_train = max(1, int(n * train_frac))
        cut_val = max(cut_train + 1, int(n * (train_frac + val_frac)))
        return cut_train, cut_val


def _side_ev_at_bar(
    candles: pd.DataFrame,
    atr: np.ndarray,
    vol_scale: np.ndarray,
    funding: np.ndarray,
    i: int,
    cfg: LabelConfig,
) -> tuple[float, float]:
    close = candles["close"].to_numpy(float)
    entry = close[i]
    vs = float(vol_scale[i])
    fr = float(funding[i]) if funding is not None else 0.0
    costs = CostModel(CostConfig())
    u_tp = entry + atr[i] * cfg.tp_mult * vs
    d_sl = entry - atr[i] * cfg.sl_mult * vs
    d_tp = entry - atr[i] * cfg.tp_mult * vs
    u_sl = entry + atr[i] * cfg.sl_mult * vs
    long_ev = _net_reward_at_barrier(entry, u_tp, d_sl, Side.LONG, costs, cfg.account_equity_usd, fr)
    short_ev = _net_reward_at_barrier(entry, d_tp, u_sl, Side.SHORT, costs, cfg.account_equity_usd, fr)
    return long_ev, short_ev


def collect_specialist_dataset(
    sym_data: dict,
    settings,
    tab_predictor,
    seq_module,
    seq_mean,
    seq_std,
    seq_device,
    seq_window,
    rl_model,
    layer_mask: dict[str, bool],
    *,
    label_cfg: LabelConfig | None = None,
    side_configs: SideLabelConfigPair | None = None,
    min_vol_z: float | None = None,
    use_extended_features: bool = True,
    tb_horizon: int = 24,
    z_window: int = 100,
    mtf_15m: dict[str, dict[str, np.ndarray]] | None = None,
) -> SpecialistDataset:
    from ae_brain.training.meta_series import rl_series, seq_series

    label_cfg = label_cfg or LabelConfig(
        tp_mult=settings.risk.atr_tp_mult,
        sl_mult=settings.risk.atr_sl_mult,
        horizon=tb_horizon,
    )
    if side_configs is None:
        side_configs = SideLabelConfigPair(
            long=SideLabelConfig(
                tp_mult=label_cfg.tp_mult,
                sl_mult=label_cfg.sl_mult,
                horizon=label_cfg.horizon,
                min_net_reward_usd=label_cfg.min_net_reward_usd,
                min_vol_z=min_vol_z,
            ),
            short=SideLabelConfig(
                tp_mult=label_cfg.tp_mult,
                sl_mult=label_cfg.sl_mult,
                horizon=label_cfg.horizon,
                min_net_reward_usd=label_cfg.min_net_reward_usd,
                min_vol_z=min_vol_z,
            ),
        )
    row_min_vol_z = min_vol_z if min_vol_z is not None else side_configs.row_min_vol_z()
    tb_horizon = max(tb_horizon, side_configs.max_horizon())

    btc_sd = sym_data.get("BTCUSDT")
    btc_ctx: dict[str, np.ndarray] | None = None
    if btc_sd is not None:
        btc_feats = btc_sd.feats
        btc_ctx = {
            "btc_ret_15": btc_feats["ret_15"].to_numpy(float),
            "btc_vol_z": btc_feats["vol_z"].to_numpy(float),
            "btc_regime_trend": btc_sd.regime_oh[:, 0],
        }

    Fs, y_long, y_short = [], [], []
    ev_long, ev_short = [], []
    symbols, regime_ids, timestamps = [], [], []

    for sym, sd in sym_data.items():
        n = len(sd.X)
        start = max(seq_window, z_window)
        end = n - tb_horizon
        if end <= start:
            continue
        funding = None
        if "funding_rate" in sd.df.columns:
            funding = pd.to_numeric(sd.df["funding_rate"], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            funding = np.zeros(n)
        if "mark_close" in sd.df.columns and "index_close" in sd.df.columns:
            mark = pd.to_numeric(sd.df["mark_close"], errors="coerce")
            index = pd.to_numeric(sd.df["index_close"], errors="coerce").replace(0, np.nan)
            sd.df = sd.df.copy()
            sd.df["basis_pct"] = ((mark - index) / index).fillna(0.0)
            sd.feats = sd.feats.copy()
            sd.feats["basis_pct"] = sd.df["basis_pct"].to_numpy(float)

        labels_long, labels_short = side_specific_profitable_labels(
            sd.df,
            sd.atr,
            long_cfg=side_configs.long.to_label_config(),
            short_cfg=side_configs.short.to_label_config(),
            vol_scale=sd.vol_scale,
            funding=funding,
        )
        tab_p = tab_predictor._calibrator.predict_proba(sd.X[:, tab_predictor._kept_idx])[:, 1]
        p_cont, trend = seq_series(seq_module, seq_mean, seq_std, sd.chan_frame, seq_window, seq_device)
        rl_expo = rl_series(rl_model, sd.X, sd.atr, sd.prices)
        high = sd.df["high"].to_numpy(float)
        low = sd.df["low"].to_numpy(float)
        extra = augment_symbol_frame(sd.feats.to_numpy(float), sd.prices, sd.atr, high, low)
        sym_mtf = (mtf_15m or {}).get(sym, {})
        if sym_mtf:
            extra = {**extra, **sym_mtf}
        vol_z = sd.feats["vol_z"].to_numpy(float)
        ts = pd.to_datetime(sd.df["ts"] if "ts" in sd.df.columns else sd.df["timestamp"], utc=True)

        for i in range(start, end):
            if row_min_vol_z is not None and vol_z[i] < row_min_vol_z:
                continue
            if use_extended_features:
                row = build_specialist_feature_row(
                    tab_p_up=tab_p[i],
                    seq_p_cont=p_cont[i],
                    seq_trend=trend[i],
                    rl_expo=rl_expo[i],
                    regime_oh=sd.regime_oh[i],
                    tabular_row=sd.feats.iloc[i].to_numpy(float),
                    extra=extra,
                    i=i,
                    symbol=sym,
                    btc_ctx=btc_ctx,
                    layer_mask=layer_mask,
                )
            else:
                row = build_meta_features(
                    tab_p[i], p_cont[i], trend[i], rl_expo[i], sd.regime_oh[i], layer_mask=layer_mask
                )
            le, se = _side_ev_at_bar(
                sd.df, sd.atr, sd.vol_scale, funding, i, side_configs.long.to_label_config()
            )
            _, se_short = _side_ev_at_bar(
                sd.df, sd.atr, sd.vol_scale, funding, i, side_configs.short.to_label_config()
            )
            se = se_short
            Fs.append(row)
            y_long.append(int(labels_long[i]))
            y_short.append(int(labels_short[i]))
            ev_long.append(le)
            ev_short.append(se)
            symbols.append(sym)
            regime_ids.append(int(np.argmax(sd.regime_oh[i])))
            timestamps.append(ts.iloc[i])

    return SpecialistDataset(
        F=np.asarray(Fs, dtype=np.float32),
        y_long=np.asarray(y_long, dtype=np.int64),
        y_short=np.asarray(y_short, dtype=np.int64),
        ev_long=np.asarray(ev_long, dtype=float),
        ev_short=np.asarray(ev_short, dtype=float),
        symbols=np.asarray(symbols),
        regime_ids=np.asarray(regime_ids, dtype=int),
        timestamps=np.asarray(timestamps),
        label_config={
            "side_configs": side_configs.to_dict(),
            "tp_mult": label_cfg.tp_mult,
            "sl_mult": label_cfg.sl_mult,
            "horizon": label_cfg.horizon,
            "min_net_reward_usd": label_cfg.min_net_reward_usd,
            "min_vol_z": row_min_vol_z,
            "use_extended_features": use_extended_features,
        },
    )
