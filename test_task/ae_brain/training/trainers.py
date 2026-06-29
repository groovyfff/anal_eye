"""High-level training entrypoints for each layer.

These tie dataset construction to the layer model objects and persist artifacts
to ``ModelConfig.artifacts_dir``. They are intentionally dependency-lazy so the
package imports cleanly even when torch / sb3 are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ae_brain.config import Settings
from ae_brain.layers.tabular import TabularPredictor
from ae_brain.training.dataset import build_sequence_dataset, build_tabular_dataset
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.train")


def train_tabular(candles: pd.DataFrame, settings: Settings) -> dict:
    """Train + calibrate the tabular layer and persist artifacts."""
    X, y, names = build_tabular_dataset(
        candles,
        tp_mult=settings.risk.atr_tp_mult,
        sl_mult=settings.risk.atr_sl_mult,
    )
    log.info("train.tabular.dataset", n=len(X), pos_rate=float(np.mean(y)))
    model = TabularPredictor(settings.model)
    metrics = model.train(X, y, feature_names=names)
    model.save(settings.model.artifacts_dir)
    return metrics


def train_sequence(candles: pd.DataFrame, settings: Settings, epochs: int = 5) -> dict:
    """Train the torch sequence model and persist weights + norm stats + ONNX."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from ae_brain.layers.sequence import SequencePredictor

    X, yc, ys, (mean, std) = build_sequence_dataset(
        candles, window=settings.model.sequence_window
    )
    log.info("train.sequence.dataset", n=len(X), cont_rate=float(np.mean(yc)))

    predictor = SequencePredictor(settings.model, settings.gpu)
    module = predictor.build_module()
    device = "cuda" if torch.cuda.is_available() and settings.gpu.enabled else "cpu"
    module = module.to(device).train()

    ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(yc), torch.from_numpy(ys)
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    last = {}
    for epoch in range(epochs):
        total = 0.0
        for xb, ycb, ysb in loader:
            xb, ycb, ysb = xb.to(device), ycb.to(device), ysb.to(device)
            opt.zero_grad()
            cont_logit, sign = module(xb)
            loss = bce(cont_logit, ycb) + 0.5 * mse(sign, ysb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        last = {"epoch": epoch, "loss": total / max(len(loader), 1)}
        log.info("train.sequence.epoch", **last)

    artifacts: Path = settings.model.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), artifacts / "sequence_model.pt")
    np.savez(artifacts / "sequence_norm.npz", mean=mean, std=std)
    log.info("train.sequence.saved", dir=str(artifacts))

    # Best-effort ONNX export for P100 fp16 serving.
    try:
        predictor.load(artifacts)
        predictor.export_onnx(artifacts)
    except Exception as exc:  # pragma: no cover
        log.warning("train.sequence.onnx_export_failed", err=str(exc))
    return last


def train_rl(candles: pd.DataFrame, settings: Settings, total_timesteps: int = 50_000) -> dict:
    """Train the RL risk agent (PPO/SAC) on a vectorized TradingEnv."""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ae_brain.features.engineering import FeatureEngineer
    from ae_brain.features.schema import FEATURE_NAMES
    from ae_brain.rl.environment import EnvConfig, TradingEnv

    eng = FeatureEngineer()
    feats = eng.compute_frame(candles)
    X = feats[list(FEATURE_NAMES)].to_numpy(np.float32)
    prices = candles["close"].to_numpy(float)
    atr = feats["atr_14"].to_numpy(float)
    atr = np.where(atr <= 0, prices * 0.005, atr)

    env_cfg = EnvConfig(feature_dim=X.shape[1])

    def _make_env():
        return TradingEnv(
            X, prices, atr,
            risk_cfg=settings.risk,
            cost_cfg=settings.cost,
            env_cfg=env_cfg,
        )

    venv = DummyVecEnv([_make_env])

    if settings.model.rl_algo == "ppo":
        model = PPO("MlpPolicy", venv, verbose=0, device="auto", n_steps=1024, batch_size=256)
    else:
        model = SAC("MlpPolicy", venv, verbose=0, device="auto")

    model.learn(total_timesteps=total_timesteps)
    artifacts = settings.model.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    model.save(str(artifacts / "rl_policy.zip"))
    log.info("train.rl.saved", algo=settings.model.rl_algo, steps=total_timesteps)
    return {"algo": settings.model.rl_algo, "timesteps": total_timesteps}
