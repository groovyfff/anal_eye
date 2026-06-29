"""Custom gymnasium trading environment for the RL risk engine.

Design principles (mandated by spec):

* **No hardcoded stop-losses.** The agent controls *exposure* directly via a
  continuous action; risk is shaped purely through the PnL reward and a
  drawdown/correlation penalty, never a fixed ``-5%`` rule.
* **Reward = net real PnL.** Each step's reward is the mark-to-market PnL of the
  held position *minus* the fees / funding / slippage incurred by changing that
  position. Costs come from the shared :class:`CostModel`, so the env and the
  live EV gate price trades identically.
* **Dynamic position sizing.** The continuous action in ``[-1, 1]`` is the
  target signed exposure as a fraction of equity; the env applies a fractional
  Kelly / volatility cap so the policy learns sizing, not just direction.
* **Correlation limit.** A penalty term discourages stacking exposure that is
  highly correlated with an externally supplied "portfolio exposure" signal.

Observation = [canonical features..., current_position, unrealized_pnl_frac,
volatility, correlated_exposure]. Action = Box([-1], [1]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False

from ae_brain.config import CostConfig, RiskConfig
from ae_brain.contracts import Side
from ae_brain.risk.costs import CostModel

_GymEnvBase = gym.Env if _HAS_GYM else object


@dataclass(slots=True)
class EnvConfig:
    feature_dim: int = 64
    episode_len: int = 512
    max_leverage: float = 5.0
    # extra appended state channels beyond raw features
    n_state_extra: int = 4
    turnover_penalty: float = 0.0005   # discourage churn
    drawdown_penalty: float = 1.5      # penalize equity drawdown (heavy)
    dd_increment_penalty: float = 5.0  # punish *new* drawdown each step (sharp)
    correlation_penalty: float = 0.25  # penalize correlated overexposure
    funding_hours_per_step: float = 8.0 / 12.0  # ~5m bars: 1/12 of an 8h epoch
    # --- risk-adjusted shaping: reward consistency, not raw PnL --------------
    sharpe_coeff: float = 0.10         # bonus on rolling Sharpe of step returns
    sortino_coeff: float = 0.15        # bonus on rolling Sortino (downside risk)
    risk_window: int = 64              # lookback for rolling Sharpe/Sortino
    risk_bonus_clip: float = 3.0       # clip risk ratios to keep reward bounded


class TradingEnv(_GymEnvBase):
    """Single-asset continuous-control trading environment.

    Parameters
    ----------
    features:
        ``(T, feature_dim)`` array of (already engineered) features per step.
    prices:
        ``(T,)`` close prices aligned with ``features``.
    atr:
        ``(T,)`` ATR aligned with ``features`` (for volatility scaling).
    correlated_exposure:
        Optional ``(T,)`` external portfolio exposure correlated with this asset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        atr: np.ndarray,
        *,
        risk_cfg: RiskConfig,
        cost_cfg: CostConfig,
        env_cfg: Optional[EnvConfig] = None,
        correlated_exposure: Optional[np.ndarray] = None,
    ) -> None:
        if not _HAS_GYM:  # pragma: no cover
            raise RuntimeError("gymnasium is required to build TradingEnv")
        super().__init__()
        self._features = np.asarray(features, dtype=np.float32)
        self._prices = np.asarray(prices, dtype=np.float64)
        self._atr = np.asarray(atr, dtype=np.float64)
        self._corr = (
            np.asarray(correlated_exposure, dtype=np.float64)
            if correlated_exposure is not None
            else np.zeros(len(prices))
        )
        self._risk = risk_cfg
        self._costs = CostModel(cost_cfg)
        self._cfg = env_cfg or EnvConfig(feature_dim=self._features.shape[1])

        obs_dim = self._features.shape[1] + self._cfg.n_state_extra
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Continuous signed target exposure in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t = 0
        self._start = 0
        self._position = 0.0          # signed exposure fraction of equity
        self._equity = 1.0            # normalized equity (starts at 1.0)
        self._peak_equity = 1.0
        self.reset()

    # ------------------------------------------------------------------ #
    def _volatility_scaled_target(self, action: float, t: int) -> float:
        """Map raw action [-1,1] to a vol-targeted exposure fraction."""
        atr_pct = self._atr[t] / self._prices[t] if self._prices[t] > 0 else 0.0
        cap = self._risk.max_position_pct
        if atr_pct > 0:
            cap = min(cap, 0.01 / atr_pct)  # ~1% equity risk per ATR
        return float(np.clip(action, -1.0, 1.0)) * cap

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        max_start = max(1, len(self._prices) - self._cfg.episode_len - 1)
        self._start = int(self.np_random.integers(0, max_start)) if _HAS_GYM else 0
        self._t = self._start
        self._position = 0.0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._prev_drawdown = 0.0
        self._returns: list[float] = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        f = self._features[self._t]
        atr_pct = self._atr[self._t] / self._prices[self._t] if self._prices[self._t] > 0 else 0.0
        extra = np.array(
            [self._position, self._equity - 1.0, atr_pct, self._corr[self._t]],
            dtype=np.float32,
        )
        return np.concatenate([f, extra]).astype(np.float32)

    def step(self, action):  # type: ignore[override]
        t = self._t
        price_now = self._prices[t]
        price_next = self._prices[min(t + 1, len(self._prices) - 1)]
        ret = (price_next - price_now) / price_now if price_now > 0 else 0.0

        target = self._volatility_scaled_target(float(np.asarray(action).reshape(-1)[0]), t)
        # Notional traded to move from current position to target.
        delta = target - self._position
        side = Side.LONG if target >= 0 else Side.SHORT
        trade_notional = abs(delta) * self._risk.account_equity_usd
        hold_notional = abs(target) * self._risk.account_equity_usd

        # --- net real PnL components (USD), normalized by equity ----------
        gross_pnl_usd = target * ret * self._risk.account_equity_usd
        costs = self._costs.estimate(
            trade_notional,
            side,
            holding_hours=self._cfg.funding_hours_per_step,
        ) if trade_notional > 0 else None
        funding_only = self._costs.funding(
            hold_notional, side, holding_hours=self._cfg.funding_hours_per_step
        )
        cost_usd = (costs.fee_usd + costs.slippage_usd if costs else 0.0) + funding_only
        net_pnl_usd = gross_pnl_usd - cost_usd
        net_pnl_frac = net_pnl_usd / self._risk.account_equity_usd

        # --- reward shaping ----------------------------------------------
        reward = net_pnl_frac
        reward -= self._cfg.turnover_penalty * abs(delta)
        # correlation penalty: punish exposure aligned with correlated book
        corr_overlap = abs(target) * abs(self._corr[t])
        if corr_overlap > self._risk.max_correlated_exposure:
            reward -= self._cfg.correlation_penalty * (corr_overlap - self._risk.max_correlated_exposure)

        # equity + drawdown (penalize both the *level* and any *new* drawdown)
        self._equity *= (1.0 + net_pnl_frac)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (self._peak_equity - self._equity) / self._peak_equity
        reward -= self._cfg.drawdown_penalty * drawdown
        dd_increment = max(0.0, drawdown - self._prev_drawdown)
        reward -= self._cfg.dd_increment_penalty * dd_increment
        self._prev_drawdown = drawdown

        # risk-adjusted shaping: reward *consistent* risk-adjusted returns
        # (rolling Sharpe + downside-only Sortino) rather than raw PnL spikes.
        self._returns.append(net_pnl_frac)
        if len(self._returns) >= 8:
            r = np.asarray(self._returns[-self._cfg.risk_window :], dtype=np.float64)
            std = float(r.std()) + 1e-8
            downside = r[r < 0.0]
            dstd = float(downside.std()) + 1e-8 if downside.size else 1e-8
            mean = float(r.mean())
            clip = self._cfg.risk_bonus_clip
            sharpe = float(np.clip(mean / std, -clip, clip))
            sortino = float(np.clip(mean / dstd, -clip, clip))
            reward += self._cfg.sharpe_coeff * sharpe + self._cfg.sortino_coeff * sortino

        self._position = target
        self._t += 1
        terminated = bool(self._equity <= 0.5)  # ruin guard (not a price stop)
        truncated = bool(self._t - self._start >= self._cfg.episode_len) or self._t >= len(self._prices) - 1

        info = {
            "equity": self._equity,
            "net_pnl_usd": net_pnl_usd,
            "cost_usd": cost_usd,
            "position": self._position,
            "drawdown": drawdown,
        }
        return self._obs(), float(reward), terminated, truncated, info
