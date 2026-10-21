# ToyStoryAI — образ приложения.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg нужен озвучке сказки: Gemini TTS отдаёт PCM, а в MAX уходит mp3.
# Ставится до копирования кода, чтобы слой переиспользовался при правках.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости — так кэш слоёв переиспользуется при правках кода.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Затем исходный код.
COPY app ./app

# Каталоги для рабочих артефактов (перекрываются томами).
RUN mkdir -p /app/uploads /app/data

CMD ["python", "-m", "app.main"]
