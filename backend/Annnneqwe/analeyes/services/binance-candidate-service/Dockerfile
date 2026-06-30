FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY services/binance-candidate-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
RUN pip install --no-cache-dir -e ./shared

COPY services/binance-candidate-service/src ./src

CMD ["python", "-m", "src.main"]
