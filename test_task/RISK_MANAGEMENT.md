# RISK_MANAGEMENT.md — Expected-Value Math & Sizing Rules

This document specifies the mathematics that govern **every** trade A.E. Brain
emits. The primary KPI is **positive Expected Value (EV) in USD, net of all
transaction costs**. Win-rate (target 55–60%) is a secondary floor only.

---

## 1. The EV Gate (the single hard guardrail)

No `LONG`/`SHORT` is ever published unless it passes the gate. The formula is
enforced verbatim in [`ae_brain/risk/ev_gate.py`](ae_brain/risk/ev_gate.py):

```python
expected_value = (prob_tp * net_reward) - (prob_sl * net_risk)
is_positive_ev = expected_value > 0
```

### Term definitions

| Term         | Meaning |
|--------------|---------|
| `prob_tp`    | Calibrated probability the **take-profit** barrier is hit *before* the stop (first-passage). |
| `prob_sl`    | Calibrated probability the **stop-loss** barrier is hit *before* the TP. |
| `gross_reward` | `|TP − entry| × qty` (USD), where `qty = notional / entry`. |
| `gross_risk`   | `|entry − SL| × qty` (USD). |
| `total_cost`   | `fees + funding + slippage` (USD, round-trip). |
| `net_reward` | `gross_reward − total_cost`. |
| `net_risk`   | `gross_risk + total_cost`. |

Costs are applied **asymmetrically and conservatively**: they *reduce* the
reward and *increase* the risk, so EV is genuinely net. A trade with raw edge
that is eaten by fees/funding/slippage is correctly rejected.

### Probability hygiene

`prob_tp` and `prob_sl` are **competing first-passage risks**, so we clamp each
to `[0,1]` and, if `prob_tp + prob_sl > 1`, renormalize proportionally. The
residual mass `1 − prob_tp − prob_sl` represents "neither barrier hit within the
horizon" (timeout) and contributes ~0 EV (closed near entry, minus costs).

A configurable **noise floor** (`AEB_FUSION_MIN_EV_USD`) can require EV to clear
a positive threshold rather than merely `> 0`, to avoid trading on estimation
noise.

### Why calibration is non-negotiable

The gate multiplies a probability by a USD amount. A model that says `0.70` but
is really `0.55` manufactures phantom positive-EV trades. We therefore force
**isotonic regression** (default) or **Platt/sigmoid scaling** on the tabular
layer, fit on a **held-out** calibration fold (see `DECISIONS.md`). Brier score
and log-loss are tracked at train time.

---

## 2. Transaction Cost Model

Implemented in [`ae_brain/risk/costs.py`](ae_brain/risk/costs.py). All outputs
are in USD so they drop straight into the EV gate.

1. **Exchange fees** — `notional × fee_rate × 2 legs`. Default taker = 4 bps,
   maker = 2 bps (Binance USD-M).
2. **Funding** — perpetual funding prorated over the expected holding period:
   `funding = notional × signed_rate × (holding_hours / 8)`. Longs pay positive
   funding; shorts receive it (and vice-versa).
3. **Slippage** — square-root market-impact model:
   `slippage = notional × (base_bps + impact_coeff·√participation) / 1e4`, applied
   on both entry and exit, where `participation = notional / ADV`.

### 2.1 Multi-asset cost handling (PRD/ТЗ #2)

The cost model is asset-class aware via a single input: the **signed funding
rate** sourced from the candidate's microstructure. Funding is a
perpetual-derivatives concept, so it only applies to the `crypto` asset class.

| Asset class | Derivative? | Funding term | Notes |
|-------------|-------------|--------------|-------|
| `crypto`    | yes | `notional × signed_rate × holding/8` | Full funding/OI/CVD/liquidation microstructure present. |
| `stock`, `metal`, `forex` | no | **forced to `0.0`** | Backend publishes funding/OI/CVD/liquidations as `null`. |

Non-crypto candidates arrive with the derivatives-only fields (`funding_rate`,
`open_interest`, `taker_buy_volume`/`cvd`, `long/short_liq_notional`, `basis`)
set to `null`. These are mapped to **neutral numeric fallbacks** at one
choke-point (`_coerce_series` in `features/engineering.py`, and `_last_float` in
`inference/engine.py`), and funding is explicitly pinned to `0.0` for
non-derivative assets. Consequently the EV gate never sees a phantom funding
cost/credit on a stock or FX trade, while **fees and slippage still apply
unchanged** to every asset class. The ~60-feature contract and the EV math are
identical across assets — only the inputs that don't exist for a given class are
neutralized.

---

## 3. Dynamic Position Sizing — **no hardcoded stops**

Implemented in [`ae_brain/risk/sizing.py`](ae_brain/risk/sizing.py).

### 3.1 ATR-based stops (volatility-adaptive)

Stops/targets are **multiples of current ATR**, never a fixed `−5%`:

```
stop_distance = ATR × atr_sl_mult      (default 1.5)
tp_distance   = ATR × atr_tp_mult      (default 2.5)
TP = entry ± tp_distance,   SL = entry ∓ stop_distance
```

The reward:risk ratio fed to Kelly is `atr_tp_mult / atr_sl_mult`.

### 3.2 Fractional Kelly stake

Full Kelly for binary payoff odds `b`:

```
f* = p − (1 − p) / b           (clamped at 0 — never size into negative edge)
```

We bet a **fraction** of Kelly (`kelly_fraction`, default 0.25) to control
variance and estimation error, then apply a **volatility target** cap so a
1-ATR adverse move risks ≈ 1% of equity:

```
f_vol_cap   = 0.01 / atr_pct
f_final     = min(f* × kelly_fraction, f_vol_cap, max_position_pct)
```

### 3.3 Leverage

Leverage is derived, not guessed: notional is sized so the stop-distance loss
≈ allocated margin (risk parity at the stop), capped by `max_leverage`:

```
notional = min(margin / (stop_distance/entry),  margin × max_leverage)
leverage = clamp(notional / margin, 0, max_leverage)
```

### 3.4 Correlation limit (portfolio risk)

Summed `|correlation|` exposure across already-open, correlated positions is
read from the `asset_correlations` snapshot. As exposure approaches the budget
(`max_correlated_exposure`), size is scaled down linearly; once the budget is
exhausted the trade is **rejected** (`correlation_budget_exhausted`). This
prevents stacking five "different" longs that are really one BTC-beta bet.

---

## 4. RL Reward = Net Real PnL

The RL risk agent ([`ae_brain/rl/environment.py`](ae_brain/rl/environment.py))
is trained with reward = **net realized PnL** per step:

```
reward = (target_exposure × return × equity − fees − funding − slippage) / equity
         − turnover_penalty·|Δposition|
         − correlation_penalty·max(0, correlated_overlap − budget)
         − drawdown_penalty·drawdown
```

There is **no price-based stop in the environment** — the only terminal "stop"
is a ruin guard at 50% equity. Risk is expressed entirely through sizing the
agent learns to manage.

---

## 5. Decision Priority (Fusion Layer)

A signal becomes `SKIP` if **any** of these fail, checked in order
([`ae_brain/layers/fusion.py`](ae_brain/layers/fusion.py)):

1. `conviction < min_conviction` (default 0.55) → SKIP
2. `position_size_pct == 0` or sizing rejected (e.g. correlation) → SKIP
3. `not is_positive_ev` → SKIP
4. otherwise → `LONG` / `SHORT`

This ordering guarantees the EV gate is the final, decisive check.

---

## 6. Audit trail of the EV decision (hybrid UPDATE/INSERT)

Every evaluated candidate — **including `SKIP`** — persists its full EV math to
`signal_feature_logs`: the calibrated layer probabilities, `gross/net` reward &
risk, `total_cost` breakdown, the final `expected_value`, the `kelly_fraction`
and derived leverage/TP/SL, and a `metrics` JSONB with the per-component
conviction breakdown, tagged with `asset_class` and `evaluated_at`.

Two write paths share the same column set (keyed on `signal_log_db_id`):

* **`signal_log_db_id > 0`** → the backend already INSERTed the row with the raw
  inputs; the ensemble **UPDATEs** that row with the EV result (no duplicate
  audit row). This is the production path.
* **`0` / missing / null** → **INSERT** fallback for local/dev, the HTTP API,
  and legacy crypto producers (fully backward compatible — a missing id never
  raises).

This makes every published *and* rejected decision reproducible: you can replay
why the gate fired (or didn't) for any asset class straight from the row.
