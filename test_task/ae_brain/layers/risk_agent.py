"""Layer 3 - RL Risk Engine inference wrapper.

Wraps a trained stable-baselines3 policy (PPO or SAC) and exposes a synchronous
``predict`` that maps a feature vector (+ portfolio state) to a *signed target
exposure* in ``[-1, 1]`` plus the critic's state-value estimate.

The agent never emits a hardcoded stop; it expresses risk preference purely as
position size/direction. Sizing constraints (Kelly, vol-target, correlation)
are applied downstream by :class:`~ae_brain.risk.sizing.PositionSizer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ae_brain.config import ModelConfig
from ae_brain.layers.base import BasePredictor
from ae_brain.utils.logging import get_logger

log = get_logger("ae_brain.risk_agent")

_POLICY_FILE = "rl_policy.zip"


@dataclass(slots=True)
class RiskAgentPrediction:
    target_exposure: float  # signed, [-1, 1]
    state_value: float


class RiskAgent(BasePredictor):
    name = "rl"

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._policy: Any = None

    def _algo_cls(self):
        if self._cfg.rl_algo == "ppo":
            from stable_baselines3 import PPO

            return PPO
        from stable_baselines3 import SAC

        return SAC

    def load(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / _POLICY_FILE
        if not path.exists():
            log.warning("rl.no_policy", dir=str(artifacts_dir))
            return
        self._policy = self._algo_cls().load(str(path), device="auto")
        log.info("rl.loaded", algo=self._cfg.rl_algo)

    def is_ready(self) -> bool:
        return self._policy is not None

    def _state_value(self, obs: np.ndarray) -> float:
        """Best-effort critic value (algo-specific; returns 0 if unavailable)."""
        try:
            import torch

            policy = self._policy.policy
            t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=policy.device)
            with torch.no_grad():
                if hasattr(policy, "predict_values"):  # PPO actor-critic
                    return float(policy.predict_values(t).cpu().item())
        except Exception:  # pragma: no cover - critic introspection is optional
            return 0.0
        return 0.0

    def predict(self, observation: np.ndarray) -> RiskAgentPrediction:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if not self.is_ready():
            return RiskAgentPrediction(target_exposure=0.0, state_value=0.0)
        action, _ = self._policy.predict(obs, deterministic=True)
        target = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        return RiskAgentPrediction(target_exposure=target, state_value=self._state_value(obs))
