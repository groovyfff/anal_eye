# Binance Candidate Service

`binance-candidate-service` converts Binance Futures **1h kline** data into neutral AE Brain candidate JSON and publishes to RabbitMQ routing key `data.candidates.ai`.

## Direct flow (no tracker-service)

```
Binance WSS 1h klines
  → binance-candidate-service (features + candidate JSON)
  → RabbitMQ data.candidates.ai
  → AE Brain (test_task/ae_brain)
  → RabbitMQ signal.final
  → notification-service
  → Telegram
```

Tracker-service is **not** required for this path.

## Why WebSocket for live updates

Binance pushes real-time kline updates as the 1h candle forms. AE Brain needs fresh `current_price` and an updated candle window without REST polling every second (rate limits / IP ban risk).

## Why one-time REST bootstrap is allowed

WebSocket streams only provide the **current** forming candle live. On startup the service fetches historical candles **once** via:

`GET /fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=200`

This seeds a rolling buffer (RSI, EMA200, MACD, etc.). After bootstrap, **all live updates come from WSS only**.

### Strictly forbidden

- REST polling every second
- Endless REST loops replacing WebSocket
- Publishing pre-made decisions (`LONG`/`SHORT`, TP/SL, leverage)

## RabbitMQ routing

| Item | Value |
|------|-------|
| vhost | `analeyes` |
| user | `analeyes` |
| exchange | `analeyes.events` |
| routing key | `data.candidates.ai` |
| queue (AE Brain) | `q_data_candidates_ai` |

Startup refuses `guest`, `/`, `%2F`, or any vhost other than `analeyes`.

## Candidate schema (neutral)

Required top-level fields: `symbol`, `asset_class`, `current_price`, `composite_score`, `features`, `candles`, `timeframe`, `market_state`.

Forbidden (never published): `decision`, `signal_type`, `side`, `heuristic_signal_consensus`, `reason_summary`, `entry_price`, `tp_price`, `sl_price`, `leverage`.

`composite_score` is a **technical strength** score (0–1), not a trade decision.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_CANDIDATE_ENABLED` | `true` | Enable service |
| `BINANCE_SYMBOLS` | `BTCUSDT` | Comma-separated symbols |
| `BINANCE_TIMEFRAME` | `1h` | Kline interval |
| `BINANCE_MARKET` | `futures` | Market label |
| `BINANCE_WSS_BASE_URL` | `wss://fstream.binance.com/ws` | WSS base (mirror: `wss://fstream.binancefuture.com/ws`) |
| `BINANCE_CANDIDATE_MIN_CANDLES` | `100` | Min history before publish (use `48` for testing) |
| `BINANCE_CANDIDATE_THROTTLE_SEC` | `60` | Max publish rate per symbol |
| `BINANCE_CANDIDATE_PUBLISH_ON_CANDLE_CLOSE` | `true` | Also publish when candle closes |
| `BINANCE_CANDIDATE_PUBLISH_ON_EVERY_UPDATE` | `false` | Publish every WSS frame |
| `BINANCE_BOOTSTRAP_LIMIT` | `200` | One-time REST kline limit |
| `BINANCE_REST_BASE_URL` | `https://fapi.binance.com` | REST bootstrap host |

## How to start

Backend:

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose up -d --build rabbitmq notification-service binance-candidate-service
```

AE Brain:

```bash
cd ~/anal_eyes/test_task
docker compose up -d --build --force-recreate ae-brain
```

## Verification

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose exec rabbitmq rabbitmqctl list_queues -p analeyes name messages consumers
docker compose exec rabbitmq rabbitmqctl list_consumers -p analeyes
docker compose logs -f binance-candidate-service notification-service
```

```bash
cd ~/anal_eyes/test_task
docker compose logs -f ae-brain
```

Expected:

- `q_data_candidates_ai` consumer=1 (AE Brain)
- `q_new_signals` consumer=1 (notification-service)
- Logs: `Published Binance candidate ... rk=data.candidates.ai`
- AE Brain: `AEBrain received candidate` → `decision=LONG/SHORT/SKIP`
- Telegram: `source_ai=ae_brain` (topic `ae_brain` in `config/settings.yml`, or `TELEGRAM_ALLOW_NO_TOPIC=true`)

## WebSocket stream

- `wss://fstream.binance.com/ws/btcusdt@kline_1h`
- Mirror if primary is silent: `BINANCE_WSS_BASE_URL=wss://fstream.binancefuture.com/ws`
