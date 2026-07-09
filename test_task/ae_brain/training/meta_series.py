"""Helpers to build per-bar sequence and RL series for meta/specialist datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ae_brain.layers.sequence import SEQ_CHANNELS


def channel_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [frame[c].to_numpy(float) if c in frame.columns else np.zeros(len(frame)) for c in SEQ_CHANNELS]
    )


def seq_series(module, mean, std, chan_frame, window, device):
    import torch

    # Memory-safe skip: when no sequence module is available (--allow-skip-sequence),
    # emit neutral 0.5 / 0.0 series so the meta-feature vector still has a valid
    # sequence column. The layer_mask downstream marks the sequence layer off.
    if module is None:
        n = len(chan_frame)
        return np.full(n, 0.5, dtype=float), np.zeros(n, dtype=float)

    chans = channel_matrix(chan_frame)
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


def rl_series(model, X, atr, prices):
    n = len(X)
    atr_pct = np.divide(atr, prices, out=np.zeros(n), where=prices > 0)
    extra = np.column_stack([np.zeros(n), np.zeros(n), atr_pct, np.zeros(n)]).astype(np.float32)
    obs = np.concatenate([X.astype(np.float32), extra], axis=1)
    act, _ = model.predict(obs, deterministic=True)
    return np.clip(np.asarray(act).reshape(-1), -1.0, 1.0)
