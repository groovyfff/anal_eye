# Интеграция external-markets-service в AnalEyes

## Структура монорепо

```
analeyes/
├── docker-compose.yml          # RabbitMQ + Postgres + 5 сервисов
├── entrypoint.sh               # pg_isready + alembic upgrade head
├── config/
│   ├── settings.yml            # market_hours, telegram.asset_class_topics
│   └── prompts/                # stock/forex/metal_gpt_prompt.txt
├── shared/                     # analeyes-shared (DB, DataEncoder, MarketHours)
└── services/
    ├── external-markets-service/
    ├── tracker-service/
    ├── ai-service/
    ├── notification-service/
    └── api-gateway/
```

## Сквозной поток (stock/forex/metal)

```
external-markets → INSERT signal_feature_logs → data.candidates.ai
       → ai-service (asset-class prompt) → signal.final
       → tracker (data.live_prices.external, session-aware FSM)
       → signal.outcome / signal.entry_event
       → notification (Telegram + OHLCV chart)
       → api-gateway /api/stats/pnl
```

## Ключевые решения

- **signal_log_db_id**: модуль пишет строку в `signal_feature_logs` до publish через `shared.database.signal_log_repository`.
- **DataEncoder**: сериализация numpy/datetime в RabbitMQ (`shared.utils.data_encoder`).
- **historical_ohlcv**: 30 свечей в payload для sequence-модели / графиков.
- **yfinance granularity**: broadcast каждые 5с, реальная свежесть котировки ~15с (`yahoo_quote_freshness_ms`), `ts` — честный Unix ms от провайдера.
- **Символы**: каноничный тикер провайдера (`GC=F`, `EURUSD=X`, `AAPL`).

## Запуск локально

```bash
cp .env.example .env
docker compose up --build
```

## Тесты

```powershell
$env:PYTHONPATH="shared/src"
pytest services/external-markets-service/tests shared/tests -q
$env:PYTHONPATH="shared/src;services/tracker-service"
pytest services/tracker-service/tests/test_session_aware_tracker.py -q
```

## Мерж в боевой монорепо

1. Скопировать `services/external-markets-service/` и изменения `shared/`.
2. Применить патчи tracker/ai/notification/api-gateway к существующим сервисам (или заменить файлами из этого scaffold).
3. Добавить сервис в корневой `docker-compose.yml` (не ломая collector/processing).
4. Прогнать alembic migration `001_add_asset_class`.

## Не в scope этого scaffold

- `collector-service` / `processing-service` (крипто-поток не менялся).
- Полный LLM-ансамбль (ai-service использует rule-based analyzer для интеграционных тестов; промпты готовы для LLM).
- testcontainers E2E (рекомендуется добавить в CI отдельным PR).
