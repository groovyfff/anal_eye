# Техническое Задание (ТЗ): Сервис мониторинга традиционных рынков
## AnalEyes Platform — External Markets Module

---

## 1. Обзор проекта и контекст

**AnalEyes** — это автоматизированная AI-торговая платформа на базе микросервисной архитектуры. Система собирает рыночные данные, рассчитывает технические индикаторы, фильтрует кандидатов и передает их на анализ нескольким AI-моделям (GPT, Gemini, Claude и др.), которые генерируют торговые сигналы (LONG/SHORT).

**В данный момент** система работает исключительно с крипторынком (Binance Futures/Spot). Все признаки, фильтры и промпты специфичны для крипты: Funding Rate, Open Interest, Liquidations, CVD и т.д.

**Цель этого ТЗ:** Разработать изолированный микросервис (`external-markets-service`), который будет подавать данные по традиционным рынкам (акции, металлы, Forex) в уже готовый пайплайн AI-анализа.

---

## 2. Архитектурная схема интеграции

```
┌───────────────────────────────────────────────────────────────────┐
│                     СУЩЕСТВУЮЩАЯ СИСТЕМА                          │
│                                                                   │
│  [Binance Collector] ──→ data.raw.binance ──→ [Processing Svc]   │
│                                                    │              │
│                                              data.candidates.ai   │
│                                                    ▼              │
│                                            [AI Service]           │
│  (NEW) [External Markets Service] ──────────────→ │              │
│            │                                       ▼              │
│            │  data.live_prices.*          [Notification Svc]      │
│            ▼                                       │              │
│  [Tracker Service (MODIFIED)] ←──────────────────────────        │
│  (слушает data.live_prices.*)                                     │
└───────────────────────────────────────────────────────────────────┘

---

## 3. External Markets Service

### 3.1 Общая задача
Написать Python-сервис, который:
1. Подключается к API поставщиков данных (Yahoo Finance, Alpha Vantage, Polygon.io или другой — на согласование).
2. Отслеживает заданный список тикеров (акции, металлы, Forex).
3. По расписанию рассчитывает набор технических индикаторов.
4. При срабатывании триггеров (интересные рыночные события) упаковывает все данные в строго определённый JSON и публикует его в RabbitMQ.
5. Непрерывно транслирует текущие цены отслеживаемых тикеров в отдельную очередь RabbitMQ.

### 3.2 Поддерживаемые типы активов и тикеры (первая версия)

| Тип | Тикеры | Источник |
|-----|--------|----------|
| Акции USA | AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META | Yahoo Finance / Polygon.io |
| Металлы | GC=F (Gold), SI=F (Silver) | Yahoo Finance |
| Индексы | ^GSPC (S&P500), ^NDX (NASDAQ) | Yahoo Finance |
| Форекс | EURUSD=X, GBPUSD=X, USDJPY=X | Yahoo Finance |

> **Примечание:** Список тикеров задаётся через конфигурационный файл и должен быть легко расширяем.

### 3.3 Таймфреймы
- Основной: `5m` (для краткосрочных сигналов)
- Дополнительный: `1h`, `4h` (для контекста тренда)
- Максимальная глубина истории в памяти: 200 свечей на каждый таймфрейм

### 3.4 Учет торговых сессий

Акции торгуются только в определённые часы. Сервис **обязан** учитывать расписание рынков:

| Рынок | Торговая сессия (UTC) |
|-------|----------------------|
| NYSE / NASDAQ | Пн–Пт, 14:30–21:00 |
| Metals (Comex) | Пн–Пт, 01:00–22:00 (с перерывами) |

- **Вне торговой сессии:** не запрашивать котировки и не генерировать сигналы.
- **Pre-market / After-hours:** опционально, можно добавить как флаг в конфиге.

### 3.5 Обязательные технические индикаторы

Для каждого тикера рассчитываются следующие метрики (используйте `pandas-ta` или `TA-Lib`):

**Трендовые:**
- EMA(8), EMA(21), EMA(50), EMA(200)
- MACD(12, 26, 9) — значения `macd_line`, `macd_signal`, `macd_hist`
- ADX(14)
- Supertrend(10, 3.0)

**Осцилляторы:**
- RSI(14)
- Stochastic(14, 3, 3)
- Bollinger Bands(20, 2) — upper, middle, lower, bandwidth

**Волатильность/Объём:**
- ATR(14) в абсолютных и процентных единицах
- VWAP (если данные о сессии есть)
- Relative Volume (объём текущей свечи / среднее за 20 аналогичных свечей)
- OBV (On-Balance Volume)

**Специфика по типу актива:**
- Акции: корреляция с S&P500 (SPY) за последние 20 периодов — `sp500_correlation`
- Металлы: корреляция с индексом доллара (DXY) за последние 20 периодов — `dxy_correlation`
- Forex: спред Bid/Ask в пунктах — `bid_ask_spread_pips`

**Уровни поддержки/сопротивления:**
- Ближайшая поддержка — `support_nearest`
- Ближайшее сопротивление — `resistance_nearest`
- Уровень VWAP текущей сессии — `vwap`

### 3.6 Логика триггеров (когда отправлять кандидата)

Сервис **не должен** спамить сигналами. Кандидат отправляется в RabbitMQ только при срабатывании хотя бы одного из триггеров:

1. **EMA Crossover:** EMA(8) пересекла EMA(21) на последней закрытой свече.
2. **RSI Extremes:** RSI вышел из зоны перепроданности (<30→>30) или перекупленности (>70→<70).
3. **Volume Spike:** `relative_volume > threshold` (по умолчанию 2.0, настраивается в конфиге).
4. **MACD Signal Cross:** `macd_hist` сменил знак (с отрицательного на положительный или наоборот).
5. **Price Breakout:** Цена пробила уровень сопротивления или поддержки, рассчитанный за последние N свечей.

> Набор и пороги триггеров должны быть вынесены в конфигурационный файл с возможностью включения/отключения каждого.

---

## 4. Протокол взаимодействия через RabbitMQ

### Подключение

- **Exchange:** `analeyes_exchange` (тип: `topic`, durable: `true`)
- **URL:** задаётся через переменную окружения `RABBITMQ_URL`

### 4.1 Очередь кандидатов для AI (`data.candidates.ai`)

Это основная очередь, из которой AI-сервис берёт данные для анализа. Ваш сервис должен отправлять JSON строго следующей схемы:

```json
{
  "signal_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "AAPL",
  "asset_class": "stock",
  "timestamp": "2024-01-15T14:30:00Z",
  "trigger_reason": "EMA_CROSSOVER_BULLISH",
  "trigger_reasons": ["EMA_CROSSOVER_BULLISH", "VOLUME_SPIKE"],
  "heuristic_signal_consensus": "LONG",
  
  "features": {
    "current_price": 182.50,
    "price_pct": 1.25,
    "market_state": "TRENDING_UP",
    
    "rsi": 55.3,
    "macd": 0.85,
    "macd_signal": 0.42,
    "macd_hist": 0.43,
    "ema_short": 181.20,
    "ema_long": 179.80,
    "ema_50": 175.00,
    "ema_200": 165.00,
    "adx": 28.5,
    "bb_upper": 185.00,
    "bb_middle": 181.00,
    "bb_lower": 177.00,
    "bb_width": 0.044,
    "atr": 2.85,
    "atr_pct": 0.0156,
    "vwap": 181.50,
    "vol_rel": 1.8,
    "obv": 123456789.0,
    
    "support_nearest": 179.00,
    "resistance_nearest": 185.50,
    
    "sp500_correlation": 0.82,
    
    "dow_monday": 0.0,
    "dow_tuesday": 0.0,
    "dow_wednesday": 1.0,
    "dow_thursday": 0.0,
    "dow_friday": 0.0,
    "dow_saturday": 0.0,
    "dow_sunday": 0.0
  },
  
  "indicators": {
    "consensus": "BULLISH",
    "consensus_strength": 0.72,
    "signals": [
      {"indicator": "ema_crossover", "signal": "BULLISH", "strength": 0.85},
      {"indicator": "macd_signals", "signal": "BULLISH", "strength": 0.65},
      {"indicator": "rsi_conditions", "signal": "NEUTRAL", "strength": 0.40}
    ]
  },
  
  "patterns": {
    "consensus": "BULLISH",
    "consensus_strength": 0.60,
    "detected_patterns_info": [
      {
        "pattern_name": "Morning Star",
        "signal": "BULLISH",
        "strength": 0.70,
        "candle_offset": 0
      }
    ]
  },
  
  "historical_snapshots": [
    {
      "timestamp": "2024-01-15T14:25:00Z",
      "close": 181.80,
      "volume": 1250000,
      "rsi": 52.1,
      "vol_rel": 1.2,
      "macd_hist": 0.18
    },
    {
      "timestamp": "2024-01-15T14:20:00Z",
      "close": 181.20,
      "volume": 980000,
      "rsi": 49.5,
      "vol_rel": 0.9,
      "macd_hist": -0.05
    }
  ],
  
  "composite_score": 0.68,
  "entry_price_suggestion": "market",
  "signal_log_db_id": null
}
```

**Обязательные поля:**
- `signal_id` — UUID v4, уникальный для каждого кандидата
- `symbol` — тикер (напр. `AAPL`, `XAUUSD`)
- `asset_class` — строго одно из: `stock`, `metal`, `forex`
- `timestamp` — ISO 8601 в UTC
- `trigger_reason` — основная причина триггера (первая в списке причин)
- `heuristic_signal_consensus` — `LONG`, `SHORT` или `NEUTRAL` (ваша оценка направления на основе триггеров)
- `features.current_price` — текущая цена актива
- `features.rsi` — значение RSI
- `features.macd_hist` — гистограмма MACD

**Дополнительно (рекомендуется):**
- `trigger_reasons` — полный список причин срабатывания в порядке детекта. Для обратной совместимости `trigger_reason` должен совпадать с `trigger_reasons[0]`.

**Описание поля `composite_score`:**  
Ваша собственная скоринговая оценка "интересности" этого кандидата в диапазоне [0.0, 1.0]. Минимальный порог для передачи в AI устанавливается конфигурацией системы. Рекомендуется считать его как взвешенную сумму нормализованных метрик (например: `vol_rel × 0.3 + rsi_score × 0.2 + ...`).

### 4.2 Очередь трансляции цен (`data.live_prices.external`)

Этот канал нужен для Tracker Service, чтобы он знал текущую цену актива при отслеживании открытых сделок. Публикуйте сообщение каждые **5 секунд** для каждого отслеживаемого тикера **во время торговой сессии**.

```json
{
  "symbol": "AAPL",
  "asset_class": "stock",
  "price": 182.75,
  "bid": 182.73,
  "ask": 182.77,
  "timestamp": "2024-01-15T14:30:05Z",
  "ts": 1705327805000
}
```

> **Важно:** Поле `ts` — это Unix Timestamp в миллисекундах. Tracker Service проверяет "свежесть" котировки. Котировки старше **4500ms** будут игнорироваться.

---

## 5. Структура проекта и требования к реализации

### 5.1 Структура каталогов (рекомендуемая)

```
external-markets-service/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── config/
│   └── settings.yml
├── src/
│   ├── main.py
│   ├── collectors/
│   │   ├── base_collector.py
│   │   ├── yahoo_finance_collector.py
│   │   └── polygon_collector.py
│   ├── logic/
│   │   ├── feature_generator.py
│   │   ├── trigger_engine.py
│   │   ├── market_hours.py
│   │   └── composite_score.py
│   └── utils/
│       ├── logger.py
│       └── pika_client.py
└── README.md
```

### 5.2 `pika_client.py`

Для удобства передаётся реализация RabbitMQ клиента из текущей системы. Вы можете использовать его как основу или написать свой — важно соответствие контракту.

**Минимальные методы:**
- `async def connect() -> bool`
- `def publish(exchange_name, routing_key, body: str)`
- `async def close()`

### 5.3 Конфигурационный файл `settings.yml`

```yaml
rabbitmq:
  url: "amqp://user:password@localhost:5672/"
  exchange: "analeyes_exchange"

data_provider:
  name: "yahoo_finance"   # Или "polygon", "alpha_vantage"
  api_key: ""             # Если требуется

watchlist:
  stocks:
    - symbol: "AAPL"
      name: "Apple Inc."
      exchange: "NASDAQ"
    - symbol: "NVDA"
      name: "NVIDIA Corporation"
      exchange: "NASDAQ"
  metals:
    - symbol: "GC=F"
      name: "Gold Futures"
    - symbol: "SI=F"
      name: "Silver Futures"
  forex:
    - symbol: "EURUSD=X"
      name: "EUR/USD"
  indices:
    - symbol: "^GSPC"
      name: "S&P 500"
      use_for_correlation: true  # Этот индекс используется только для расчёта корреляции, не генерирует сигналы

main_timeframe: "5m"
context_timeframes:
  - "1h"
  - "4h"
history_depth: 200

scan_interval_s: 60          # Как часто пересчитывать индикаторы
price_broadcast_interval_s: 5 # Как часто транслировать цены

market_hours:
  timezone: "America/New_York"
  stock_open: "09:30"
  stock_close: "16:00"
  pre_market_enabled: false
  after_hours_enabled: false

triggers:
  ema_crossover:
    enabled: true
    fast_period: 8
    slow_period: 21
  rsi_extreme:
    enabled: true
    oversold_exit: 30
    overbought_exit: 70
  volume_spike:
    enabled: true
    threshold: 2.0
    window: 20
  macd_signal_cross:
    enabled: true
  price_breakout:
    enabled: true
    lookback_periods: 20

composite_score:
  weights:
    vol_rel: 0.30
    rsi_score: 0.20
    macd_hist_score: 0.20
    adx_score: 0.15
    pattern_score: 0.15

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 5.4 Переменные окружения `.env.example`

```ini
RABBITMQ_URL=amqp://user:password@rabbitmq:5672/
DATA_PROVIDER_API_KEY=your_api_key_here
```

### 5.5 `docker-compose.yml` для тестирования

```yaml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password

  external-markets-service:
    build: .
    env_file: .env
    depends_on:
      - rabbitmq
    volumes:
      - ./config:/app/config
      - ./src:/app/src
```

---

## 6. Нефункциональные требования

### 6.1 Rate Limiting и отказоустойчивость
- Реализовать задержки между запросами к API поставщика в соответствии с его лимитами.
- При получении HTTP 429 (Too Many Requests) — экспоненциальный backoff (1s, 2s, 4s, максимум 5 попыток).
- Логировать все ошибки API с уровнем ERROR.

### 6.2 Обработка данных
- Все вычисления через `pandas` + `numpy`.
- Для индикаторов использовать `pandas-ta` (предпочтительно) или `TA-Lib==0.4.28`.
- Если недостаточно данных для расчёта индикатора (менее требуемого периода свечей), поле должно быть `null` в JSON, а не вызывать исключение.

### 6.3 Logging
- Структурированное логирование в stdout.
- Уровни: DEBUG (детальная отладка), INFO (нормальный режим), WARNING (нестандартные ситуации), ERROR (ошибки).
- Каждое сообщение должно содержать `[SERVICE_NAME]` и, где уместно, `[SYMBOL]`.

### 6.4 Тестирование
- Покрыть unit-тестами:
  - `trigger_engine.py` — логику срабатывания каждого триггера.
  - `feature_generator.py` — корректность расчёта хотя бы RSI, MACD, EMA.
  - `market_hours.py` — определение торговых часов.
