"""A.E. Brain - Autonomous 4-layer predictive trading ensemble for Binance.

Layers
------
1. Tabular Predictor  (LightGBM / XGBoost / CatBoost, calibrated probabilities)
2. Sequence Predictor (LSTM / GRU / PatchTST, trend continuation/reversal)
3. Risk Engine        (PPO / SAC RL agent over a custom gymnasium env)
4. Fusion / Output    (EV-gated deterministic LONG / SHORT / SKIP decision)

The system is fully asynchronous (asyncio) and offloads heavy model inference
to thread/process pools. All trades must pass the Expected-Value gate before
being published to ``signal.final``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
