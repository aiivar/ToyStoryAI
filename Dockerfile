# ToyStoryAI — образ приложения.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Сначала зависимости — так кэш слоёв переиспользуется при правках кода.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Затем исходный код.
COPY app ./app

# Каталоги для рабочих артефактов (перекрываются томами).
RUN mkdir -p /app/uploads /app/data

CMD ["python", "-m", "app.main"]
