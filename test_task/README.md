# A.E. Brain — Autonomous Predictive Trading Ensemble for Binance

A standalone, fully-asynchronous, **4-layer machine-learning ensemble** that
turns market candidates into strict, EV-gated `LONG` / `SHORT` / `SKIP` trading
signals. **No external LLM APIs** are used in the decision path — it is a pure
numerical / mathematical ML pipeline.

> Primary KPI: **positive Expected Value (EV) in USD, net of fees, funding and
> slippage.** Win-rate (55–60%) is a secondary floor.

```
data.candidates.ai ──▶ Feature Engineering (TA-Lib, ~60 feats)
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                            ▼
  ① Tabular         ② Sequence                  ③ RL Risk Engine
  LightGBM/XGB/Cat  LSTM/GRU/PatchTST           PPO/SAC over gymnasium env
  (calibrated p)    (trend continuation)        (signed exposure, net-PnL reward)
        └─────────────────┼───────────────────────────┘
                          ▼
                  ④ Fusion + EV Gate
        expected_value = prob_tp·net_reward − prob_sl·net_risk
                          ▼
                    signal.final  (+ signal_feature_logs in PostgreSQL)
```

## Layers

1. **Tabular Predictor** — gradient boosting on ~60 features with **strict
   probability calibration** (isotonic / Platt). → `prob_tp`.
2. **Sequence Predictor** — PatchTST / LSTM / GRU on a ≥30-candle window,
   fp16/ONNX on Tesla P100. → trend continuation/reversal probability.
3. **RL Risk Engine** — PPO/SAC over a custom `gymnasium` env; reward = **net
   real PnL**; dynamic fractional-Kelly / ATR sizing; correlation limits; **no
   hardcoded stops**.
4. **Fusion / Output** — aggregates calibrated outputs, enforces the **EV gate**,
   emits a deterministic dict: `Decision`, `position_size`, `leverage`, dynamic
   `take_profit`, dynamic `stop_loss`.

## Quickstart

```bash
pip install "numpy<2.0"           # FIRST (TA-Lib 0.4.28 ABI)
pip install -r requirements.txt && pip install -e .

ae-brain gen-data --rows 20000 --out data/candles.parquet
ae-brain train all --data data/candles.parquet
ae-brain gen-candidate --out examples/candidate.json
ae-brain evaluate --candidate examples/candidate.json
ae-brain run                      # live RabbitMQ loop
```

See **[ENVIRONMENT.md](ENVIRONMENT.md)** for full setup (P100/CUDA, PostgreSQL,
RabbitMQ), **[RISK_MANAGEMENT.md](RISK_MANAGEMENT.md)** for the EV math & sizing
rules, and **[DECISIONS.md](DECISIONS.md)** for the architectural rationale.

## Project layout

```
ae_brain/
  config.py            # pydantic settings (env-driven)
  contracts.py         # typed messages: TradeCandidate, LayerProbabilities, FinalSignal
  features/            # ~60-feature schema + TA-Lib engineering
  data/                # async PostgreSQL, candle repo, schema.sql, ChromaDB RAG
  risk/                # cost model, fractional-Kelly/ATR sizing, EV gate
  layers/              # tabular, sequence (+nets), risk_agent, fusion
  rl/                  # custom gymnasium TradingEnv
  inference/           # async engine (Thread/Process pool offload)
  messaging/           # RabbitMQ consumer/publisher (try/finally ack)
  training/            # dataset labeling, synthetic data, trainers
  runtime.py           # live wiring; api.py FastAPI; cli.py Typer CLI
tests/                 # pure-core smoke tests (no heavy deps)
```

## Tests

```bash
pytest -q   # validates EV math, sizing, feature shape, fusion JSON
```

## Disclaimer

Research/engineering scaffold. Trading derivatives with leverage is risky;
validate on out-of-sample data and paper-trade before any capital is at risk.
