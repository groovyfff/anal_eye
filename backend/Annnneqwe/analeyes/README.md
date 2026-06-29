## Файл `config/settings.yml`

### `rabbitmq`
- `url` — URL подключения к RabbitMQ.
- `exchange` — имя exchange для публикации сообщений.
- `candidate_queue` — имя очереди для кандидатов AI.
- `live_prices_queue` — имя очереди live-котировок.
- `connect_retries` — число попыток первичного подключения к RabbitMQ.
- `connect_retry_delay_s` — пауза между попытками подключения (секунды).

### `data_provider`
- `name` — поставщик данных (`yahoo_finance` или `polygon`).
- `api_key` — API-ключ провайдера (если требуется).
- `request_delay_s` — минимальная пауза между запросами к API (секунды).
- `max_retries` — максимум попыток при ошибках/лимитах API.

### `watchlist`
- `stocks` — список акций.
  - `symbol` — тикер.
  - `name` — отображаемое имя.
  - `exchange` — биржа инструмента.
- `metals` — список металлов.
  - `symbol` — тикер.
  - `name` — отображаемое имя.
- `forex` — список forex-пар.
  - `symbol` — тикер.
  - `name` — отображаемое имя.
- `indices` — список индексов/бенчмарков.
  - `symbol` — тикер.
  - `name` — отображаемое имя.
  - `use_for_correlation` — использовать ли инструмент только для корреляционных признаков.

### `main_timeframe`
- Основной таймфрейм сканирования сигналов.

### `context_timeframes`
- Список дополнительных таймфреймов для контекста тренда.

### `history_depth`
- Глубина истории свечей в памяти для расчетов.

### `scan_interval_s`
- Интервал сканирования рынка (секунды).

### `scan_workers`
- Максимальное число параллельных задач сканирования.

### `price_broadcast_interval_s`
- Интервал публикации live-котировок (секунды).

### `use_thread_pool`
- Включить выполнение блокирующих вызовов провайдера в thread pool.

### `market_hours`
- `timezone` — таймзона для интерпретации торговых сессий.
- `stock_open` — локальное время открытия рынка акций.
- `stock_close` — локальное время закрытия рынка акций.
- `use_fixed_utc_window` — использовать фиксированное UTC-окно вместо локального расписания.
- `stock_open_utc` — UTC-время открытия при фиксированном режиме.
- `stock_close_utc` — UTC-время закрытия при фиксированном режиме.
- `pre_market_enabled` — включить pre-market для акций.
- `after_hours_enabled` — включить after-hours для акций.
- `metal_breaks_utc` — список UTC-интервалов перерывов для металлов (`HH:MM-HH:MM`).

### `triggers`
- `ema_crossover`
  - `enabled` — включение триггера.
  - `fast_period` — быстрый период EMA.
  - `slow_period` — медленный период EMA.
- `rsi_extreme`
  - `enabled` — включение триггера.
  - `oversold_exit` — уровень выхода из перепроданности.
  - `overbought_exit` — уровень выхода из перекупленности.
- `volume_spike`
  - `enabled` — включение триггера.
  - `threshold` — порог относительного объема.
  - `window` — окно расчета базового объема.
- `macd_signal_cross`
  - `enabled` — включение триггера.
- `price_breakout`
  - `enabled` — включение триггера.
  - `lookback_periods` — глубина истории для расчета support/resistance.

### `composite_score`
- `min_publish_score` — минимальный composite-score для публикации кандидата.
- `weights` — веса метрик в composite-score.
  - `vol_rel`
  - `rsi_score`
  - `macd_hist_score`
  - `adx_score`
  - `pattern_score`

### `patterns`
- `enabled` — включить анализ свечных паттернов.
- `lookback_candles` — число последних свечей для детекта паттернов.
- `max_detected` — максимум паттернов в payload.
- `min_strength` — минимальная сила паттерна для учета.

### `logging`
- `service_name` — имя сервиса в логах.
- `level` — уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `json` — включить JSON-формат логов.
- `format` — строковый формат логов (используется при `json: false`).

## Файл `.env` / `.env.example`
- `RABBITMQ_URL` — переопределяет `rabbitmq.url` из `settings.yml`.
- `DATA_PROVIDER_API_KEY` — переопределяет `data_provider.api_key` из `settings.yml`.
