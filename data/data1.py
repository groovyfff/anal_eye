import json
import os

# Пути к файлам
INPUT_FILE = "raw_candles.json"
OUTPUT_FILE = "processed_candidate.json"

def convert_candles():
    # 1. Проверяем наличие сырых данных
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Ошибка: Файл '{INPUT_FILE}' не найден!")
        print("Создай его рядом со скриптом и закинь туда массив свечей.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            print("❌ Ошибка: '{INPUT_FILE}' содержит невалидный JSON!")
            return

    # Если это одиночный объект вместо списка, оборачиваем в список
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    print(f"📦 Считано свечей из исходного файла: {len(raw_data)}")

    # 2. Фильтруем поля и конвертируем строки в float
    processed_candles = []
    for index, c in enumerate(raw_data):
        try:
            clean_candle = {
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"])
            }
            processed_candles.append(clean_candle)
        except KeyError as e:
            print(f"⚠️ Пропущена свеча под индексом {index}: отсутствует поле {e}")
        except ValueError as e:
            print(f"⚠️ Пропущена свеча под индексом {index}: ошибка конвертации в число ({e})")

    # 3. Валидация под требования ML-ядра
    if len(processed_candles) < 48:
        print(f"⚠️ Внимание: У тебя всего {len(processed_candles)} свечей. Модели ae_brain нужно хотя бы 48-64 для инференса!")

    # 4. Собираем финальный контракт CandidateIn
    final_payload = {
        "symbol": "BTCUSDT",      # Можешь поменять на нужный тикер
        "interval": "1d",         # Таймфрейм твоих данных
        "signal_log_db_id": 1,
        "asset_class": "crypto",
        "candles": processed_candles,
        "meta": {
            "correlation_id": "manual-rabbitmq-test-run"
        }
    }

    # 5. Сохраняем готовый результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print(f"🚀 Успешно преобразовано свечей: {len(processed_candles)}")
    print(f"💾 Результат сохранен в: '{OUTPUT_FILE}'")
    print("\n👉 Теперь просто открой этот файл, скопируй ВСЁ содержимое и вставь в поле Payload в RabbitMQ!")

if __name__ == "__main__":
    convert_candles()

