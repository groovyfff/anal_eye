# RabbitMQ validation (AnalEyes V6 topology)

Exchange: `analeyes.events` (topic, durable)  
Vhost: `analeyes`  
App user: `analeyes`

## Candidate processing (AE Brain)

**AE Brain** (`test_task/ae_brain`) is the sole consumer of `q_data_candidates_ai`.
The legacy `ai-service` is behind Docker profile `legacy-ai` and does not consume
candidates unless `AI_SERVICE_CONSUME_CANDIDATES=true`.

| Step | Component | Queue / routing |
|------|-----------|-----------------|
| Publish | external-markets / manual test | `data.candidates.ai` → `q_data_candidates_ai` |
| Consume | AE Brain (`aeb-app`) | `q_data_candidates_ai` |
| Publish | AE Brain | `signal.final` |
| Consume | notification-service | `q_new_signals` |
| Consume | tracker-service | `q_tracker_signals` |

## Service mapping (this repo)

| V6 service | Current compose service | Role |
|---|---|---|
| processing-service | `external-markets-service` | Publishes `data.candidates.ai` |
| AI (candidates) | `test_task/ae_brain` | Consumes `q_data_candidates_ai`, publishes `signal.final` |
| AI (legacy) | `ai-service` (profile `legacy-ai`) | Disabled by default |
| notification-service | `notification-service` | Consumes `q_new_signals`, etc. |
| tracker-service | `tracker-service` | Consumes `q_tracker_signals` |

## 1. Start main backend (without legacy ai-service)

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose up -d --build
docker compose stop ai-service   # ensure legacy consumer is off
```

## 2. Start AE Brain (connects to main broker via host.docker.internal)

```bash
cd ~/anal_eyes/test_task
# Set RABBITMQ_APP_PASSWORD in .env to match main backend
docker compose up -d --build --force-recreate
docker compose logs -f ae-brain
```

## 3. RabbitMQ CLI checks

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose exec rabbitmq rabbitmqctl list_vhosts
docker compose exec rabbitmq rabbitmqctl list_queues -p analeyes name messages consumers
docker compose exec rabbitmq rabbitmqctl list_bindings -p analeyes
docker compose exec rabbitmq rabbitmqctl list_consumers -p analeyes
docker compose exec rabbitmq rabbitmqctl list_queues -p / name messages consumers
```

Expected after AE Brain starts:

- `q_data_candidates_ai` → **consumers=1** (consumer tag `ae-brain-q_data_candidates_ai`)
- `q_new_signals` → consumers=1 (notification-service)
- No consumer from `ai-service` on `q_data_candidates_ai`
- `/` vhost has no application traffic

## 4. Publish neutral test candidate (main broker)

```bash
docker compose exec rabbitmq rabbitmqadmin -u admin -p changeme -V analeyes publish \
  exchange=analeyes.events routing_key=data.candidates.ai \
  payload='{"source":"binance","market":"futures","asset_class":"crypto","symbol":"BTCUSDT","timeframe":"1m","current_price":65000.0,"market_state":"trend","composite_score":0.82,"features":{"current_price":65000.0,"rsi":55.4,"macd":1.2,"macd_signal":0.8,"macd_hist":0.4,"adx":28.0,"atr":420.0,"ema_short":64800.0,"ema_long":64200.0,"ema_50":64000.0,"ema_200":62000.0,"volume_change":0.35,"price_change_1h":2.1},"candles":[{"timestamp":1769680000000,"open":64800.0,"high":65100.0,"low":64750.0,"close":65000.0,"volume":902.43}]}'
```

Expect AE Brain logs: received → normalized → running analysis → result → (if LONG/SHORT) published `signal.final`.

## 5. Tail logs

```bash
docker compose logs -f notification-service
# from test_task:
docker compose logs -f ae-brain
```

## Acceptance

- Connections use user `analeyes` on vhost `analeyes` (not `guest` / `/`)
- `q_data_candidates_ai` consumed only by AE Brain
- `q_new_signals` consumed by notification-service
- No external LLM in AE Brain decision path
