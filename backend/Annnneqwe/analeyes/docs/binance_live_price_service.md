# Binance Live Price Service

`binance-live-price-service` connects to **Binance Futures WebSocket** kline streams and publishes normalized live crypto prices to RabbitMQ for `tracker-service`.

## What it does

1. Subscribes to Binance Futures kline streams (default: `btcusdt@kline_1h`).
2. On **every** kline update for the currently forming candle, reads the kline **close** field as the live price.
3. Publishes a JSON message to RabbitMQ routing key `data.live_prices.external`.
4. Optionally publishes normalized/raw kline payloads to `data.raw.binance`.

It does **not** generate trading candidates, call AE Brain, or publish `signal.final`.

## Why WebSocket (not REST polling)

AE Brain works on **1h** candles. Binance pushes real-time kline updates over WebSocket as the candle forms. A REST loop polling every second would:

- Hit Binance rate limits and risk IP bans
- Waste bandwidth and add latency

This service uses **only** Binance Futures WebSocket streams (`wss://fstream.binance.com`).

## RabbitMQ routing

| Item | Value |
|------|-------|
| vhost | `analeyes` |
| user | `analeyes` |
| exchange | `analeyes.events` (durable topic) |
| live price routing key | `data.live_prices.external` |
| live price queue (consumer) | `q_data_live_prices_external` |
| optional raw kline routing key | `data.raw.binance` |

Startup refuses `guest`, `/`, `%2F`, or any vhost other than `analeyes`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_LIVE_ENABLED` | `true` | Set `false` to idle without connecting |
| `BINANCE_SYMBOLS` | `BTCUSDT` | Comma-separated symbols (uppercase in output) |
| `BINANCE_TIMEFRAME` | `1h` | Kline interval (`kline_1h` stream suffix) |
| `BINANCE_MARKET` | `futures` | Market label in payloads |
| `BINANCE_WSS_BASE_URL` | `wss://fstream.binance.com/ws` | Single-stream base URL |
| `BINANCE_RECONNECT_DELAY_SEC` | `5` | Delay before WSS reconnect |
| `BINANCE_PUBLISH_RAW_KLINE` | `true` | Also publish to `data.raw.binance` |
| `BINANCE_LOG_EVERY_N` | `20` | Throttle per-price logs |
| `RABBITMQ_URL` | (from split vars) | AMQP URL; must use vhost `analeyes` |

## How to start

```bash
cd ~/anal_eyes/backend/Annnneqwe/analeyes
docker compose up -d --build binance-live-price-service tracker-service notification-service rabbitmq
```

## How to verify

```bash
docker compose logs -f binance-live-price-service tracker-service
```

RabbitMQ:

```bash
docker compose exec rabbitmq rabbitmqctl list_queues -p analeyes name messages consumers
docker compose exec rabbitmq rabbitmqctl list_bindings -p analeyes
docker compose exec rabbitmq rabbitmqctl list_consumers -p analeyes
```

Expected:

- `q_data_live_prices_external` has **1 consumer** (`tracker-service`)
- `binance-live-price-service` logs `Published Binance live price symbol=BTCUSDT ...`
- `tracker-service` accepts messages with `symbol`, `price`, `ts`, `asset_class=crypto`

Sample live-price payload:

```json
{
  "source": "binance",
  "market": "futures",
  "symbol": "BTCUSDT",
  "asset_class": "crypto",
  "price": 81396.61,
  "bid": null,
  "ask": null,
  "ts": 1778795999000,
  "timestamp": "2026-06-29T22:00:00Z",
  "timeframe": "1h",
  "candle_open_time": 1778792400000,
  "candle_close_time": 1778795999000,
  "is_candle_closed": false,
  "raw_stream": "btcusdt@kline_1h"
}
```

## How tracker-service consumes output

`tracker-service` binds `q_data_live_prices_external` to `data.live_prices.external`. On each message it calls `ExternalPriceStore.upsert_external_message()`, which requires `symbol`, `price`, and a fresh `ts` (within ~5s). The check loop builds a `market_data_map` and evaluates active BTCUSDT signals for entry / TP / SL.

## WebSocket URL

- Single symbol: `wss://fstream.binance.com/ws/btcusdt@kline_1h`
- Multiple symbols: `wss://fstream.binance.com/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h`

If `fstream.binance.com` connects but no kline messages arrive (network/region), try the futures mirror:

`BINANCE_WSS_BASE_URL=wss://fstream.binancefuture.com/ws`
