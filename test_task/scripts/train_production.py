"""Production training orchestrator for the A.E. Brain ensemble.

Trains the full production-quant stack on a deep, multi-symbol dataset:

    0. Regime model   - unsupervised GaussianMixture over volatility+momentum,
                        injected as a one-hot into tabular + sequence features.
    1. Tabular        - LightGBM (walk-forward CV + early stopping) on the full
                        66-feature vector, then **SHAP-pruned** (drop bottom 20%)
                        and retrained; kept feature list -> features_schema.json.
    2. Sequence       - PatchTST with per-bar regime channels + head dropout.
    3. RL risk agent  - PPO across one env per symbol; drawdown + Sharpe/Sortino
                        shaped reward.
    4. Meta-model     - multinomial stacker over the 3 base outputs + regime,
                        emitting LONG/SHORT/SKIP (replaces the heuristic EV gate).

All artifacts are written to ``ModelConfig.artifacts_dir`` with the filenames the
inference container mounts. Feature engineering is computed **once per symbol**
and reused across every layer (the rolling Hurst/autocorr applies are the slow
part), so the whole pipeline stays tractable on CPU.

Usage::

    python scripts/train_production.py --data-dir data/production \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT --seq-epochs 4 --rl-timesteps 150000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ae_brain.config import Settings
from ae_brain.features.engineering import FeatureEngineer
from ae_brain.features.regime import REGIME_INPUT_FEATURES, RegimeModel
from ae_brain.features.schema import FEATURE_NAMES, REGIME_ONEHOT_NAMES
from ae_brain.layers.meta import MetaModel, build_meta_features
from ae_brain.layers.sequence import SEQ_CHANNELS, SequencePredictor
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.training.dataset import (
    directional_barrier_labels,
    relative_vol_scale,
    triple_barrier_labels,
)
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.train_production")

_FEATURES_SCHEMA_FILE = "features_schema.json"
_Z_WINDOW = 100
_HORIZON = 12  # sequence forward horizon (also used for barrier label horizon)
_TB_HORIZON = 24  # triple-barrier horizon for tabular/meta labels


# --------------------------------------------------------------------------- #
@dataclass
class SymbolData:
    df: pd.DataFrame
    feats: pd.DataFrame      # full 66-feature frame (incl. regime one-hot)
    X: np.ndarray            # feats[FEATURE_NAMES] as float32 (N, 66)
    prices: np.ndarray
    atr: np.ndarray
    vol_scale: np.ndarray
    regime_oh: np.ndarray    # (N, 3)
    chan_frame: pd.DataFrame  # df augmented with regime columns (for SEQ_CHANNELS)


def _load_symbol_frames(data_dir: Path, symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = data_dir / f"{sym}_{interval}.csv"
        if not path.exists():
            log.warning("train.production.missing_csv", path=str(path))
            continue
        df = pd.read_csv(path)
        if "ts" in df:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        frames[sym] = df
        log.info("train.production.loaded", symbol=sym, rows=len(df))
    if not frames:
        raise SystemExit(f"no per-symbol CSVs found under {data_dir} for {symbols}")
    return frames


# --------------------------------------------------------------------------- #
# Phase 0: regime model + per-symbol feature cache
# --------------------------------------------------------------------------- #
def fit_regime_and_features(
    frames: dict[str, pd.DataFrame], settings: Settings
) -> tuple[RegimeModel, dict[str, SymbolData]]:
    # 1) Base features (no regime) - the expensive step, done once per symbol.
    base_eng = FeatureEngineer(z_window=_Z_WINDOW)
    base: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        base[sym] = base_eng.compute_frame(df)
        log.info("train.production.features", symbol=sym, rows=len(base[sym]))

    # 2) Fit the regime model on the pooled volatility+momentum descriptors.
    pooled = pd.concat([b[list(REGIME_INPUT_FEATURES)] for b in base.values()], ignore_index=True)
    regime = RegimeModel(n_regimes=settings.model.n_regimes).fit(pooled)
    regime.save(settings.model.artifacts_dir)
    counts = np.bincount(regime.predict(pooled), minlength=settings.model.n_regimes)
    log.info("train.production.regime.fit", counts={i: int(c) for i, c in enumerate(counts)})

    # 3) Augment cached features with the regime one-hot (no recompute needed).
    sym_data: dict[str, SymbolData] = {}
    for sym, df in frames.items():
        feats = base[sym].copy()
        oh = regime.predict_one_hot(feats)
        for j, name in enumerate(REGIME_ONEHOT_NAMES):
            feats[name] = oh[:, j]
        prices = df["close"].to_numpy(float)
        atr = feats["atr_14"].to_numpy(float)
        atr = np.where(atr <= 0, prices * 0.005, atr)
        vol_scale = relative_vol_scale(feats["atr_pct"].to_numpy(float))
        chan_frame = df.copy()
        for j, name in enumerate(REGIME_ONEHOT_NAMES):
            chan_frame[name] = oh[:, j]
        sym_data[sym] = SymbolData(
            df=df,
            feats=feats,
            X=feats[list(FEATURE_NAMES)].to_numpy(np.float32),
            prices=prices,
            atr=atr,
            vol_scale=vol_scale,
            regime_oh=oh,
            chan_frame=chan_frame,
        )
    return regime, sym_data


# --------------------------------------------------------------------------- #
# Tabular (+ SHAP pruning)
# --------------------------------------------------------------------------- #
def _interleave(arrays: list[np.ndarray]) -> np.ndarray:
    n = min(len(a) for a in arrays)
    cols = arrays[0].shape[1] if arrays[0].ndim > 1 else None
    head = np.stack([a[:n] for a in arrays], axis=1).reshape(
        (n * len(arrays),) + ((cols,) if cols else ())
    )
    tails = [a[n:] for a in arrays if len(a) > n]
    return np.concatenate([head, *tails], axis=0) if tails else head


def _shap_keep(model, X: np.ndarray, keep_frac: float, sample: int = 6000) -> list[int]:
    """Return indices of the top ``keep_frac`` features by mean |SHAP|."""
    import shap

    rng = np.random.default_rng(7)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[idx])
    if isinstance(sv, list):  # older shap: [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    importance = np.abs(sv).mean(axis=0)
    n_keep = max(1, int(round(len(importance) * keep_frac)))
    kept = sorted(np.argsort(importance)[::-1][:n_keep].tolist())
    return kept


def train_tabular_multi(sym_data: dict[str, SymbolData], settings: Settings) -> dict:
    xs, ys = [], []
    for sym, sd in sym_data.items():
        y = triple_barrier_labels(
            sd.df, sd.atr,
            tp_mult=settings.risk.atr_tp_mult, sl_mult=settings.risk.atr_sl_mult,
            horizon=_TB_HORIZON, vol_scale=sd.vol_scale,
        )
        valid = slice(_Z_WINDOW, len(sd.X) - _TB_HORIZON)
        xs.append(sd.X[valid])
        ys.append(y[valid])
    X = _interleave(xs)
    y = _interleave([yi.reshape(-1, 1) for yi in ys]).reshape(-1).astype(np.int64)
    log.info("train.production.tabular.dataset", n=len(X), n_features=X.shape[1], pos_rate=float(np.mean(y)))

    # 1) Full-feature fit (for SHAP importances).
    full = TabularPredictor(settings.model)
    full_metrics = full.train(X, y, feature_names=list(FEATURE_NAMES))

    # 2) SHAP prune bottom 20%, retrain on the kept subset.
    try:
        kept_idx = _shap_keep(full._model, X, keep_frac=0.80)
    except Exception as exc:  # pragma: no cover - shap optional/edge
        log.warning("train.production.shap_failed", err=str(exc))
        kept_idx = list(range(X.shape[1]))
    kept_names = [FEATURE_NAMES[i] for i in kept_idx]

    pruned = TabularPredictor(settings.model)
    pruned_metrics = pruned.train(X[:, kept_idx], y, feature_names=kept_names)
    pruned.save(settings.model.artifacts_dir)

    # 3) Persist the authoritative feature schema for inference.
    schema = {
        "all_features": list(FEATURE_NAMES),
        "kept_features": kept_names,
        "n_all": len(FEATURE_NAMES),
        "n_kept": len(kept_names),
        "pruned_features": [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in set(kept_idx)],
        "regime_input_features": list(REGIME_INPUT_FEATURES),
        "regime_onehot": list(REGIME_ONEHOT_NAMES),
        "n_regimes": settings.model.n_regimes,
        "seq_channels": list(SEQ_CHANNELS),
    }
    (settings.model.artifacts_dir / _FEATURES_SCHEMA_FILE).write_text(json.dumps(schema, indent=2))

    return {
        "full_auc": full_metrics["auc"],
        "full_cv_auc": full_metrics["cv_auc_mean"],
        "pruned_auc": pruned_metrics["auc"],
        "pruned_cv_auc": pruned_metrics["cv_auc_mean"],
        "pruned_precision": pruned_metrics["precision"],
        "n_features_kept": len(kept_names),
        "n_features_pruned": len(FEATURE_NAMES) - len(kept_names),
        "pruned_out": schema["pruned_features"],
    }


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #
def _channel_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [frame[c].to_numpy(float) if c in frame else np.zeros(len(frame)) for c in SEQ_CHANNELS]
    )


def _windows(chan_frame: pd.DataFrame, mean, std, window: int, horizon: int):
    chans = _channel_matrix(chan_frame)
    norm = (chans - mean) / np.where(std == 0, 1.0, std)
    close = chan_frame["close"].to_numpy(float)
    xs, yc, ys = [], [], []
    for i in range(window, len(chan_frame) - horizon):
        xs.append(norm[i - window : i])
        trail = (close[i - 1] - close[i - window]) / close[i - window]
        fwd = (close[i + horizon] - close[i]) / close[i]
        trend_sign = np.sign(trail)
        yc.append(1 if np.sign(fwd) == trend_sign and trend_sign != 0 else 0)
        ys.append(float(np.clip(trail * 50, -1, 1)))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(yc, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def train_sequence_multi(sym_data, settings, *, epochs, cap, batch_size):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    window = settings.model.sequence_window
    horizon = _HORIZON

    all_chans = np.concatenate([_channel_matrix(sd.chan_frame) for sd in sym_data.values()], axis=0)
    mean = all_chans.mean(axis=0)
    std = all_chans.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    xs, yc, ys = [], [], []
    for sym, sd in sym_data.items():
        x, c, s = _windows(sd.chan_frame, mean, std, window, horizon)
        xs.append(x); yc.append(c); ys.append(s)
    X = np.concatenate(xs); YC = np.concatenate(yc); YS = np.concatenate(ys)

    rng = np.random.default_rng(13)
    n = len(X)
    perm = rng.permutation(n)
    n_val = int(n * 0.15)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if cap and len(tr_idx) > cap:
        tr_idx = rng.choice(tr_idx, size=cap, replace=False)
    log.info("train.production.sequence.dataset", total=n, train=len(tr_idx), val=len(val_idx),
             channels=X.shape[-1], cont_rate=float(np.mean(YC)))

    predictor = SequencePredictor(settings.model, settings.gpu)
    module = predictor.build_module()
    device = "cuda" if torch.cuda.is_available() and settings.gpu.enabled else "cpu"
    module = module.to(device).train()

    ds = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(YC[tr_idx]), torch.from_numpy(YS[tr_idx]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    last: dict = {}
    for epoch in range(epochs):
        total = 0.0
        for xb, ycb, ysb in loader:
            xb, ycb, ysb = xb.to(device), ycb.to(device), ysb.to(device)
            opt.zero_grad()
            cont_logit, sign = module(xb)
            loss = bce(cont_logit, ycb) + 0.5 * mse(sign, ysb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        sched.step()
        last = {"epoch": epoch, "loss": total / max(len(loader), 1)}
        log.info("train.production.sequence.epoch", **last)

    module.eval()
    with torch.no_grad():
        logits, _ = module(torch.from_numpy(X[val_idx]).to(device))
        p = torch.sigmoid(logits).float().cpu().numpy()
    val_auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(YC[val_idx])) == 2:
            val_auc = float(roc_auc_score(YC[val_idx], p))
    except Exception:
        pass

    artifacts = settings.model.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), artifacts / "sequence_model.pt")
    np.savez(artifacts / "sequence_norm.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    log.info("train.production.sequence.saved", dir=str(artifacts), val_auc=val_auc)
    return ({"loss": last.get("loss"), "val_auc": val_auc, "train_windows": int(len(tr_idx))},
            module, mean, std, device, window)


# --------------------------------------------------------------------------- #
# RL
# --------------------------------------------------------------------------- #
def train_rl_multi(sym_data, settings, *, total_timesteps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ae_brain.rl.environment import EnvConfig, TradingEnv

    feature_dim = next(iter(sym_data.values())).X.shape[1]
    env_cfg = EnvConfig(feature_dim=feature_dim)

    def _make(sd: SymbolData):
        def _factory():
            return TradingEnv(sd.X, sd.prices, sd.atr, risk_cfg=settings.risk,
                              cost_cfg=settings.cost, env_cfg=env_cfg)
        return _factory

    venv = DummyVecEnv([_make(sd) for sd in sym_data.values()])
    model = PPO("MlpPolicy", venv, verbose=0, device="auto", n_steps=1024, batch_size=256,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.005, learning_rate=3e-4, n_epochs=10)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(settings.model.artifacts_dir / "rl_policy.zip"))

    eval_sd = next(iter(sym_data.values()))
    eval_env = TradingEnv(eval_sd.X, eval_sd.prices, eval_sd.atr, risk_cfg=settings.risk,
                          cost_cfg=settings.cost, env_cfg=env_cfg)
    ep_rewards = []
    for _ in range(20):
        obs, _ = eval_env.reset(); done = False; total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, te, tr, _ = eval_env.step(action)
            total += float(reward); done = bool(te or tr)
        ep_rewards.append(total)
    mean_reward = float(np.mean(ep_rewards))
    log.info("train.production.rl.saved", steps=total_timesteps, mean_episode_reward=mean_reward, n_envs=len(sym_data))
    return {"timesteps": total_timesteps, "mean_episode_reward": mean_reward,
            "reward_std": float(np.std(ep_rewards)), "n_envs": len(sym_data)}, model


# --------------------------------------------------------------------------- #
# Meta-model (stacking)
# --------------------------------------------------------------------------- #
def _seq_series(module, mean, std, chan_frame, window, device):
    import torch

    chans = _channel_matrix(chan_frame)
    norm = ((chans - mean) / np.where(std == 0, 1.0, std)).astype(np.float32)
    n = len(norm)
    p_cont = np.full(n, 0.5, dtype=float)
    trend = np.zeros(n, dtype=float)
    if n <= window:
        return p_cont, trend
    windows = np.stack([norm[i - window : i] for i in range(window, n)]).astype(np.float32)
    module.eval()
    pc, ts = [], []
    with torch.no_grad():
        for k in range(0, len(windows), 4096):
            xb = torch.from_numpy(windows[k : k + 4096]).to(device)
            cl, sg = module(xb)
            pc.append(torch.sigmoid(cl).float().cpu().numpy().reshape(-1))
            ts.append(sg.float().cpu().numpy().reshape(-1))
    p_cont[window:] = np.concatenate(pc)
    trend[window:] = np.concatenate(ts)
    return p_cont, trend


def _rl_series(model, X, atr, prices):
    n = len(X)
    atr_pct = np.divide(atr, prices, out=np.zeros(n), where=prices > 0)
    extra = np.column_stack([np.zeros(n), np.zeros(n), atr_pct, np.zeros(n)]).astype(np.float32)
    obs = np.concatenate([X.astype(np.float32), extra], axis=1)
    act, _ = model.predict(obs, deterministic=True)
    return np.clip(np.asarray(act).reshape(-1), -1.0, 1.0)


def train_meta_multi(sym_data, settings, tab_predictor, seq_module, seq_mean, seq_std,
                     seq_device, seq_window, rl_model):
    Fs, ys = [], []
    for sym, sd in sym_data.items():
        n = len(sd.X)
        start = max(seq_window, _Z_WINDOW)
        end = n - _TB_HORIZON
        if end <= start:
            continue
        labels = directional_barrier_labels(
            sd.df, sd.atr, tp_mult=settings.risk.atr_tp_mult, sl_mult=settings.risk.atr_sl_mult,
            horizon=_TB_HORIZON, vol_scale=sd.vol_scale,
        )
        tab_p = tab_predictor._calibrator.predict_proba(sd.X[:, tab_predictor._kept_idx])[:, 1]
        p_cont, trend = _seq_series(seq_module, seq_mean, seq_std, sd.chan_frame, seq_window, seq_device)
        rl_expo = _rl_series(rl_model, sd.X, sd.atr, sd.prices)
        for i in range(start, end):
            Fs.append(build_meta_features(tab_p[i], p_cont[i], trend[i], rl_expo[i], sd.regime_oh[i]))
            ys.append(int(labels[i]))
        log.info("train.production.meta.symbol", symbol=sym, n=end - start)

    F = np.asarray(Fs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    meta = MetaModel(model_kind="logreg")
    metrics = meta.fit(F, y)
    meta.save(settings.model.artifacts_dir)
    return metrics


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production A.E. Brain ensemble")
    parser.add_argument("--data-dir", type=Path, default=Path("data/production"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--seq-epochs", type=int, default=4)
    parser.add_argument("--seq-cap", type=int, default=60000)
    parser.add_argument("--seq-batch", type=int, default=256)
    parser.add_argument("--rl-timesteps", type=int, default=150_000)
    args = parser.parse_args()

    settings = Settings()
    if args.artifacts is not None:
        settings.model.artifacts_dir = args.artifacts
    settings.model.artifacts_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    frames = _load_symbol_frames(args.data_dir, symbols, args.interval)

    summary: dict = {"symbols": list(frames), "artifacts_dir": str(settings.model.artifacts_dir)}
    t0 = time.time()

    print("\n=== [0/4] Regime model + feature cache ===", flush=True)
    regime, sym_data = fit_regime_and_features(frames, settings)

    print("\n=== [1/4] Tabular (LightGBM + SHAP prune) ===", flush=True)
    summary["tabular"] = train_tabular_multi(sym_data, settings)
    print(json.dumps(summary["tabular"], indent=2, default=str), flush=True)

    print("\n=== [2/4] Sequence (PatchTST + regime channels) ===", flush=True)
    seq_metrics, seq_module, seq_mean, seq_std, seq_device, seq_window = train_sequence_multi(
        sym_data, settings, epochs=args.seq_epochs, cap=args.seq_cap, batch_size=args.seq_batch
    )
    summary["sequence"] = seq_metrics
    print(json.dumps(seq_metrics, indent=2, default=str), flush=True)

    print("\n=== [3/4] RL Risk Agent (PPO) ===", flush=True)
    rl_metrics, rl_model = train_rl_multi(sym_data, settings, total_timesteps=args.rl_timesteps)
    summary["rl"] = rl_metrics
    print(json.dumps(rl_metrics, indent=2, default=str), flush=True)

    print("\n=== [4/4] Meta-model (stacking) ===", flush=True)
    # Reload the pruned tabular predictor so meta uses the deployed kept features.
    tab_predictor = TabularPredictor(settings.model)
    tab_predictor.load(settings.model.artifacts_dir)
    summary["meta"] = train_meta_multi(
        sym_data, settings, tab_predictor, seq_module, seq_mean, seq_std, seq_device, seq_window, rl_model
    )
    print(json.dumps(summary["meta"], indent=2, default=str), flush=True)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
