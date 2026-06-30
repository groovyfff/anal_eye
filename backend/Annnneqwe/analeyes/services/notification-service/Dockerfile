FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY services/notification-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
RUN pip install --no-cache-dir -e ./shared

COPY config /app/config
COPY services/notification-service/src ./src
COPY scripts ./scripts
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "src.main"]
