FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/tracker-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared
RUN pip install -e ./shared

COPY services/tracker-service/config ./config
COPY services/tracker-service/src ./src

CMD ["python", "-m", "src.main"]
