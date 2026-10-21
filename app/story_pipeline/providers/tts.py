"""Провайдеры синтеза речи: RouterAI (Gemini TTS) и mock без ключа.

Контракт скрыт за протоколом `NarrationProvider`, чтобы этап озвучки
принимал фейк без сети. Провайдер отдаёт WAV: Gemini TTS в RouterAI умеет
только `pcm` (16 бит, 24 кГц, моно), поэтому обёртка в контейнер — работа
провайдера, а не этапа. Склейка сегментов и кодирование в mp3 живут в этапе:
там же, где известны номера страниц и пауза между ними.

Повторы относятся только к временным сбоям: таймаут, транспортная ошибка,
429 и 5xx. Отказ по ключу, модели или голосу повторять бессмысленно — это
`PermanentNarrationError`, и он сразу превращается в отсутствие озвучки.
"""

from __future__ import annotations

import io
import re
import wave
from typing import Any, Protocol

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.story_pipeline.errors import (
    PermanentNarrationError,
    TransientNarrationError,
)
from app.story_pipeline.models import Voice
from app.story_pipeline.settings import StorySettings

# Параметры потока, который отдаёт Gemini TTS через RouterAI.
SAMPLE_WIDTH = 2
CHANNELS = 1
SAMPLE_RATE = 24000

# Оценка темпа для mock-режима: русская речь — около 15 знаков в секунду.
MOCK_CHARS_PER_SECOND = 15.0

# Потолок длительности одного mock-сегмента: тишина должна подтверждать
# формат и склейку, а не занимать десятки мегабайт в каталоге прогона.
MOCK_MAX_SEGMENT_SECONDS = 3.0


class NarrationProvider(Protocol):
    """Синтез речи по тексту в WAV-контейнере."""

    async def synthesize(self, text: str) -> bytes:
        """Озвучить текст и вернуть WAV (24 кГц, моно, 16 бит)."""

    async def aclose(self) -> None:
        """Освободить сетевые ресурсы провайдера."""


def pcm_to_wav(pcm: bytes) -> bytes:
    """Обернуть сырой PCM в WAV-контейнер с параметрами Gemini TTS."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(CHANNELS)
        target.setsampwidth(SAMPLE_WIDTH)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)
    return buffer.getvalue()


class MockNarrationProvider:
    """Тишина вместо речи: без сети, без ключа и без `ffmpeg`.

    Длительность считается по длине текста, поэтому склейка страниц в тестах
    и в локальном прогоне даёт разное число секунд и пауза между страницами
    проверяема. Потолок на сегмент — только чтобы каталог `.tmp` не пух от
    десятиминутной тишины.
    """

    def __init__(self, *, chars_per_second: float = MOCK_CHARS_PER_SECOND) -> None:
        self._chars_per_second = chars_per_second

    async def synthesize(self, text: str) -> bytes:
        compact = re.sub(r"\s+", " ", text).strip()
        seconds = min(
            MOCK_MAX_SEGMENT_SECONDS,
            max(0.5, len(compact) / self._chars_per_second),
        )
        frames = int(seconds * SAMPLE_RATE)
        return pcm_to_wav(b"\x00\x00" * frames)

    async def aclose(self) -> None:
        """Mock не держит ресурсов."""


class GeminiSpeechProvider:
    """Синтез речи через OpenAI-совместимый `audio/speech` RouterAI."""

    def __init__(
        self,
        settings: StorySettings,
        voice: Voice,
        *,
        client: httpx.AsyncClient | None = None,
        max_attempts: int | None = None,
        retry_wait: Any | None = None,
    ) -> None:
        self._settings = settings
        self._voice = voice
        self._voice_name = settings.narration_voice_name(voice)
        self._client = client
        attempts = max_attempts or (settings.routerai_narration_retries + 1)
        self._max_attempts = max(1, attempts)
        self._retry_wait = (
            retry_wait
            if retry_wait is not None
            else wait_exponential(multiplier=1, min=1, max=8)
        )

    async def synthesize(self, text: str) -> bytes:
        """Озвучить текст, повторив временные сбои, и вернуть WAV."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientNarrationError),
            reraise=True,
        ):
            with attempt:
                return await self._request(text)
        raise AssertionError("tenacity не вернул результат и не бросил")  # pragma: no cover

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент, если он создан провайдером."""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.routerai_narration_timeout
            )
        return self._client

    def _payload(self, text: str) -> dict[str, Any]:
        """Тело запроса: текст дословно, манера — только в provider.options."""

        payload: dict[str, Any] = {
            "model": self._settings.routerai_narration_model,
            "input": text,
            "voice": self._voice_name,
            "response_format": "pcm",
        }
        style = self._settings.routerai_narration_style.strip()
        if style:
            payload["provider"] = {
                "options": {
                    "google-ai-studio": {
                        "speech_metadata": {"style": style}
                    }
                }
            }
        return payload

    async def _request(self, text: str) -> bytes:
        url = (
            f"{self._settings.routerai_base_url.rstrip('/')}/audio/speech"
        )
        client = self._build_client()
        try:
            response = await client.post(
                url,
                json=self._payload(text),
                headers={
                    "Authorization": f"Bearer {self._settings.routerai_api_key}"
                },
            )
        except httpx.TransportError as exc:
            raise TransientNarrationError(
                f"сеть синтеза речи недоступна: {exc.__class__.__name__}"
            ) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientNarrationError(
                f"провайдер синтеза речи ответил {response.status_code}"
            )
        if response.status_code >= 400:
            # Тело ответа провайдера в текст ошибки не попадает: в нём может
            # оказаться эхо запроса, а сообщение уходит в лог и в story.json.
            raise PermanentNarrationError(
                f"провайдер синтеза речи отклонил запрос: "
                f"{response.status_code}"
            )

        data = response.content
        if not data:
            raise TransientNarrationError("провайдер вернул пустой аудиопоток")

        logger.debug(
            "narration: {} знаков, голос {}, {} байт PCM",
            len(text),
            self._voice_name,
            len(data),
        )
        return pcm_to_wav(data)


def make_narration_provider(
    settings: StorySettings,
    voice: Voice,
) -> NarrationProvider:
    """Провайдер по настройкам: mock без ключа, иначе Gemini через RouterAI.

    Пустой ключ — это не ошибка, а режим разработки: прогон обязан
    завершаться, не тратя денег, ровно как у mock-провайдера картинок.
    """

    if not settings.routerai_api_key.strip():
        logger.warning(
            "ROUTERAI_API_KEY пуст: озвучка работает в mock-режиме (тишина)"
        )
        return MockNarrationProvider()
    return GeminiSpeechProvider(settings, voice)


__all__ = [
    "GeminiSpeechProvider",
    "MockNarrationProvider",
    "NarrationProvider",
    "make_narration_provider",
    "pcm_to_wav",
]
