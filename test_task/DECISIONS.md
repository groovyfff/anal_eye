# DECISIONS.md — ML Architecture Rationale

Why each layer is built the way it is. The north star is **net positive EV in
USD**, so every choice is justified by how it affects the inputs to the EV gate
(`prob_tp`, `prob_sl`, costs, sizing).

---

## 0. Why a 4-layer ensemble (and not one big model)

Markets are non-stationary and multi-scale. A single model that tries to learn
tabular microstructure, sequence dynamics, and risk preference at once is hard
to calibrate and impossible to debug. We separate concerns:

| Layer | Question it answers | Output consumed by EV gate |
|-------|---------------------|----------------------------|
| Tabular | "Given the current snapshot, what is P(reach +R before −R)?" | `prob_tp` |
| Sequence | "Is the prevailing trend continuing or reversing?" | refines direction / `prob_tp` |
| RL Risk | "Given my book, how much signed exposure should I take?" | sizing/direction prior |
| Fusion | "Net of costs, is this +EV? How large? What TP/SL?" | the decision |

Each layer is independently trainable, swappable, and observable.

---

## 1. Tabular Predictor — gradient boosting + calibration

* **Algorithm: LightGBM (default), XGBoost / CatBoost selectable.** GBDTs
  dominate tabular finance features: they handle heterogeneous scales, missing
  values, monotone-ish microstructure relationships, and are fast/low-latency
  on CPU (GIL-light → safe in a `ProcessPoolExecutor`). Default LightGBM for its
  speed/leaf-wise growth; XGBoost (`hist`) and CatBoost are drop-in alternatives.
* **Label = triple-barrier.** The target is *exactly* the quantity the EV gate
  needs: did price reach `+ATR·tp_mult` before `−ATR·sl_mult` within a horizon?
  Training the model on the same barrier geometry the EV gate prices removes a
  train/serve objective mismatch.
* **Calibration is mandatory.** We wrap the booster in
  `CalibratedClassifierCV` with **isotonic** (default) or **sigmoid/Platt**
  scaling, fit on a **held-out, time-ordered** fold (`cv="prefit"`). Because EV
  multiplies probability × USD, calibration error translates directly into
  mispriced EV. We monitor **Brier** + **log-loss** at train time.
* **~60 features, explicit schema.** The contract lives in
  `features/schema.py` (no auto-generation) so it is auditable and the
  train/serve column order can never drift (`assert_feature_order`). Families:
  volatility (`vol_z`, `bb_width`, ATR), momentum, **order flow** (`cvd`,
  `ofi_delta`), **derivatives** (`oi_z`, funding, basis, liquidations), volume,
  return shape, and regime/seasonality.

---

## 2. Sequence Layer — PatchTST (default), LSTM / GRU optional

* **Default = PatchTST.** Patched Transformers for time series give strong
  long-horizon accuracy with **channel-independent patch embeddings**, which is
  both statistically robust (fewer cross-channel spurious correlations) and
  cheap to run in fp16. The patching also shortens the effective sequence,
  cutting attention cost — important for low-latency P100 inference.
* **LSTM / GRU fallbacks.** Recurrent encoders remain excellent, lower-variance
  baselines on short windows and are useful when data is scarce; both are
  provided and selectable via `AEB_MODEL_SEQUENCE_BACKEND`.
* **Window ≥ 30 (default 48).** Enforced in `config.py`. The model consumes an
  OHLCV+flow channel window and emits two heads: a **continuation logit**
  (→ sigmoid → `p_continuation`) and a **trend-sign** head (`tanh`, [−1,1]).
  Splitting magnitude (sign) from probability lets the fusion layer combine
  "which way" with "how confident" cleanly.
* **fp16 / ONNX.** Modules are cast to half precision and routed round-robin
  across the 4 P100s; an ONNX export path enables ONNXRuntime-GPU serving.

---

## 3. Risk Engine — PPO (default) / SAC over a custom gymnasium env

* **Why RL for sizing, not direction.** Direction is better handled by
  supervised models with calibrated probabilities. Sizing under
  costs/funding/drawdown/correlation is a sequential decision problem with
  delayed reward — exactly what RL is good at. The agent outputs a **continuous
  signed target exposure** in `[−1,1]`.
* **PPO default; SAC optional.** PPO is stable, well-behaved with continuous
  actions, and easy to reproduce; SAC (off-policy, entropy-regularized) is
  offered for higher sample efficiency when replaying large histories.
* **Reward = net real PnL** (fees + funding + slippage subtracted), using the
  *same* `CostModel` as the live EV gate — so the policy optimizes the metric we
  actually trade on. Shaping terms penalize turnover, drawdown, and correlated
  overexposure.
* **No hardcoded stops.** The environment has no `−5%` rule; the only terminal
  guard is ruin at 50% equity. Risk is sizing, learned end-to-end.

---

## 4. Fusion / Output — deterministic, EV-gated

* **Linear opinion pool over directional signals.** Each layer is mapped to a
  signal in `[−1,1]` (`2p−1` for probabilities, signed exposure for RL) and
  blended with configurable weights. Linear pooling is transparent, monotone,
  and easy to audit/tune — preferable to an opaque meta-learner for a
  risk-critical gate (and it avoids another model to calibrate).
* **Probabilities → first-passage `prob_tp`/`prob_sl`.** The fusion layer maps
  the calibrated tabular/sequence probabilities to the competing-risk pair the
  EV gate consumes, aligned to the chosen side.
* **Strict priority gating** (conviction → sizing/correlation → **EV > 0**)
  yields a deterministic dict: `Decision`, `position_size_pct`, `leverage`,
  dynamic `take_profit`, dynamic `stop_loss`, plus the full EV breakdown for
  audit. Same input ⇒ same output (no sampling in the serving path).

---

## 5. Infrastructure choices

* **asyncio + executors.** The event loop owns I/O (RabbitMQ, PostgreSQL);
  CPU-bound feature engineering goes to a `ProcessPoolExecutor` (GIL-immune) and
  GIL-light model inference to a `ThreadPoolExecutor` (avoids per-process CUDA
  contexts and large-array IPC).
* **RabbitMQ with guaranteed ack.** `try/finally` around every handler prevents
  consumer starvation / queue overflow; poison messages are dead-lettered.
* **PostgreSQL audit-first.** Every decision (incl. `SKIP`) and its features are
  logged to `signal_feature_logs`, enabling offline calibration monitoring and
  realized-vs-predicted EV backtests — the feedback loop for retraining.
* **No external LLM APIs.** Pure numerical/ML pipeline end-to-end. ChromaDB RAG
  is optional, off by default, and never in the decision path.
* **`numpy < 2.0`.** Required for TA-Lib 0.4.28 ABI compatibility; pinned
  everywhere.

## 6. News sentiment features — RabbitMQ-only, echo-only

* **Boundary contract.** AE Brain receives news sentiment **strictly through
  RabbitMQ** (`data.news.sentiment` / `q_data_news_sentiment`), published by the
  separate `news-sentiment-service`. There are no direct imports, calls, or
  shared memory between the two services. `messaging/news_features.py` is the
  sole integration point and is fully isolated (its own aio-pika connection,
  channel, and queue).
* **Echo-only by design (for now).** AE Brain re-derives all scoring features
  from candles, so the latest fresh news snapshot is attached to
  `candidate.meta["news"]` (and mirrored under `candidate.meta["features"]` with
  `news_` prefixes) for **logging/echo only**. It does NOT yet change trade
  decisions. Wiring news into `FusionContext` (mirroring `btc_specialist_ctx`)
  is the documented next step.
* **Graceful degradation.** The consumer is gated behind
  `enable_news_features` (default `false`), so existing deployments are
  unaffected. Snapshots expire after `news_features_max_age_s` (default 300s);
  when no fresh snapshot exists for a symbol, the candidate is untouched and
  scoring proceeds normally. Invalid payloads are ACKed (no poison loop).

