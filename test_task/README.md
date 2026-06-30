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

## RabbitMQ migration (AE Brain = sole candidate consumer)

AE Brain connects to the **main** AnalEyes broker (`host.docker.internal:5672`, vhost `analeyes`).
The legacy `ai-service` in the main backend is behind profile `legacy-ai` and does not
consume `q_data_candidates_ai` by default.

```bash
# Main backend (no legacy ai-service)
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose up -d --build
docker compose stop ai-service

# AE Brain (set RABBITMQ_APP_PASSWORD in test_task/.env to match main .env)
cd ~/anal_eyes/test_task
docker compose up -d --build --force-recreate
docker compose logs -f ae-brain
```

Verify consumers on main broker:

```bash
docker compose exec rabbitmq rabbitmqctl list_queues -p analeyes name messages consumers
docker compose exec rabbitmq rabbitmqctl list_consumers -p analeyes
```

Expected: `q_data_candidates_ai` → `consumers=1`, tag `ae-brain-q_data_candidates_ai`.

Publish neutral candidate to **main** broker:

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose exec rabbitmq rabbitmqadmin -u admin -p changeme -V analeyes publish \
  exchange=analeyes.events routing_key=data.candidates.ai \
  payload='{"source":"binance","market":"futures","asset_class":"crypto","symbol":"BTCUSDT","timeframe":"1m","current_price":65000.0,"market_state":"trend","composite_score":0.82,"features":{"current_price":65000.0,"rsi":55.4,"macd":1.2,"macd_signal":0.8,"macd_hist":0.4,"adx":28.0,"atr":420.0,"ema_short":64800.0,"ema_long":64200.0,"ema_50":64000.0,"ema_200":62000.0,"volume_change":0.35,"price_change_1h":2.1},"candles":[{"timestamp":1769680000000,"open":64800.0,"high":65100.0,"low":64750.0,"close":65000.0,"volume":902.43}]}'
```

See also [backend docs/rabbitmq_validation.md](../backend/Annnneqwe/analeyes/docs/rabbitmq_validation.md).

## RabbitMQ end-to-end (local broker profile `local-mq`)

AE Brain consumes neutral candidates from `analeyes.events` / `data.candidates.ai`
(queue `q_data_candidates_ai`) and publishes `signal.final` on the same exchange.
No external LLM APIs are used — decisions come from the internal fusion engine only.

```bash
cd test_task
docker compose up -d --build --force-recreate
docker compose --profile app up -d --build --force-recreate

# Follow AE Brain logs
docker compose logs -f ae-brain

# Publish a neutral candidate (no decision field) to the local broker
docker compose exec rabbitmq rabbitmqadmin -u admin -p changeme -V analeyes publish \
  exchange=analeyes.events routing_key=data.candidates.ai \
  payload='{"source":"binance","market":"futures","asset_class":"crypto","symbol":"BTCUSDT","timeframe":"1m","current_price":65000.0,"market_state":"trend","composite_score":0.82,"features":{"current_price":65000.0,"rsi":55.4,"macd":1.2,"macd_signal":0.8,"macd_hist":0.4,"adx":28.0,"atr":420.0,"ema_short":64800.0,"ema_long":64200.0,"ema_50":64000.0,"ema_200":62000.0,"volume_change":0.35,"price_change_1h":2.1},"candles":[{"timestamp":1769680000000,"open":64800.0,"high":65100.0,"low":64750.0,"close":65000.0,"volume":902.43},{"timestamp":1769680060000,"open":65000.0,"high":65200.0,"low":64900.0,"close":65120.0,"volume":840.12}]}'

# Peek one queued message without consuming it
docker compose run --rm --no-deps --entrypoint python3 ae-brain -m ae_brain.tools.dump_one_candidate
```

To reach the **main** AnalEyes notification-service, set in `.env`:

```bash
AEB_OUTPUT_AMQP_URL=amqp://analeyes:<password>@host.docker.internal:5672/analeyes
AEB_INPUT_AMQP_URL=amqp://analeyes:<password>@host.docker.internal:5672/analeyes
```

Main broker verification:

```bash
docker compose exec rabbitmq rabbitmqctl list_vhosts
docker compose exec rabbitmq rabbitmqctl list_queues -p analeyes name messages consumers
docker compose exec rabbitmq rabbitmqctl list_bindings -p analeyes
docker compose exec rabbitmq rabbitmqctl list_queues -p / name messages consumers
```


```bash
pytest -q   # validates EV math, sizing, feature shape, fusion JSON
```

## Disclaimer

Research/engineering scaffold. Trading derivatives with leverage is risky;
validate on out-of-sample data and paper-trade before any capital is at risk.
