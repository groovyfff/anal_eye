import pika
import json
import os

# --- 1. НАСТРОЙКИ ---
# Путь к твоему живому файлу, выкачанному через binance_data_exporter
REAL_DATA_FILE = "raw_candles.json" 

# Новые учетные данные, которые ты только что создал
RABBIT_USER = "dev"
RABBIT_PASS = "dev"
RABBIT_HOST = "localhost"
RABBIT_PORT = 5672

# --- 2. ЧТЕНИЕ И ПОДГОТОВКА ДАННЫХ ---
if not os.path.exists(REAL_DATA_FILE):
    print(f"❌ Ошибка: Файл '{REAL_DATA_FILE}' не найден! Убедись, что он лежит в этой же папке.")
    exit(1)

try:
    with open(REAL_DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except Exception as e:
    print(f"❌ Ошибка парсинга JSON: {e}")
    exit(1)

# Достаем массив свечей
candles_list = raw_data.get("candles", raw_data) if isinstance(raw_data, dict) else raw_data

# Отрезаем 200 последних живых свечей для нейронки
recent_candles = candles_list[-200:]

# Собираем идеальный контракт
payload = {
    "symbol": "BTCUSDT",
    "interval": "1h", 
    "signal_log_db_id": 9999,
    "asset_class": "crypto",
    "meta": {
        "correlation_id": "real-market-python-test",
        "adv_usd": 1500000000.0,
        "expected_holding_hours": 8.0,
        "current_position": 0.0,
        "correlated_exposure": 0.0
    },
    "candles": recent_candles
}

# --- 3. ОТПРАВКА В RABBITMQ ---
print(f"🚀 Подключаемся к RabbitMQ на {RABBIT_HOST}:{RABBIT_PORT} под пользователем '{RABBIT_USER}'...")

try:
    credentials = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBIT_HOST, 
        port=RABBIT_PORT, 
        virtual_host='/', 
        credentials=credentials
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Публикуем сообщение в нужную очередь
    channel.basic_publish(
        exchange='',
        routing_key='data.candidates.ai',
        body=json.dumps(payload),
        properties=pika.BasicProperties(
            delivery_mode=2, # Сделать сообщение персистентным (сохранять на диск)
            content_type='application/json'
        )
    )

    print(f"✅ УСПЕХ! {len(recent_candles)} реальных рыночных свечей успешно залетели в Кролика.")
    print("👉 Открывай логи Докера: docker compose logs -f ae-brain")
    
    connection.close()

except pika.exceptions.AMQPConnectionError as e:
    print(f"❌ Ошибка подключения к RabbitMQ: {e}")
except Exception as e:
    print(f"❌ Неизвестная ошибка: {e}")
