"""Провайдеры генерации изображений: Gemini и детерминированный mock.

Контракт скрыт за протоколом `ImageProvider`, чтобы этап иллюстрирования
принимал фейк без сети. `GeminiImageProvider` изолирует вендор-специфику
`generateContent` (ключ в заголовке, base64-части, защитный разбор ответа);
`MockImageProvider` строит валидный PNG из seed стандартной библиотекой и
включается при пустом `gemini_api_key`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
import zlib
from typing import Any, Protocol

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    TransientImageGenerationError,
)
from app.story_pipeline.settings import StorySettings

# Сигнатура PNG-файла: первые восемь байт любого PNG.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Сторона квадратного mock-изображения в пикселях. Небольшой размер
# достаточен, чтобы файл был валидным PNG, и не раздувает тесты и вывод.
_MOCK_IMAGE_SIZE = 32

# Тип цвета PNG: truecolor (по три байта на пиксель, без альфа-канала).
_PNG_COLOR_TYPE_RGB = 2

# Глубина цвета PNG в битах на канал.
_PNG_BIT_DEPTH = 8

# MIME reference-картинок: character sheet и страницы сохраняются как PNG.
_REFERENCE_MIME_TYPE = "image/png"

# Запрошенные модальности ответа Gemini: только изображение.
_RESPONSE_MODALITIES = ["IMAGE"]

# HTTP-статусы Gemini, при которых повтор бессмыслен: аутентификация и
# валидация запроса.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


class GeneratedImage(BaseModel):
    """Результат генерации: байты изображения и использованный seed."""

    data: bytes
    seed: int | None = None


class ImageProvider(Protocol):
    """Контракт генератора изображений: промпт → байты картинки."""

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        """Сгенерировать изображение по промпту и reference-картинкам."""

        ...


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Собрать PNG-чанк: длина, тег, данные и CRC32 по тегу с данными."""

    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _mock_png(seed: int) -> bytes:
    """Построить валидный PNG, детерминированно зависящий от seed.

    Пиксели заполняются потоком байтов из SHA-256 от seed: одинаковый seed
    даёт одинаковые байты, разные seed (с точностью до коллизии хеша) —
    разные. Только стандартная библиотека, без внешних зависимостей.
    """

    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    pixels = bytearray()
    for row in range(_MOCK_IMAGE_SIZE):
        # Байт фильтра строки: 0 — без фильтрации, как требует формат.
        pixels.append(0)
        for column in range(_MOCK_IMAGE_SIZE):
            offset = (row * _MOCK_IMAGE_SIZE + column) * 3
            pixels.extend(
                digest[(offset + channel) % len(digest)]
                for channel in range(3)
            )

    header = struct.pack(
        ">IIBBBBB",
        _MOCK_IMAGE_SIZE,
        _MOCK_IMAGE_SIZE,
        _PNG_BIT_DEPTH,
        _PNG_COLOR_TYPE_RGB,
        0,  # метод сжатия
        0,  # метод фильтрации
        0,  # без черезстрочной развёртки
    )
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(pixels)))
        + _png_chunk(b"IEND", b"")
    )


class MockImageProvider:
    """Офлайн-провайдер: детерминированный PNG из seed, без сети."""

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        """Вернуть PNG, выведенный из seed; промпт и референсы не нужны."""

        effective_seed = 0 if seed is None else seed
        return GeneratedImage(data=_mock_png(effective_seed), seed=seed)


class GeminiImageProvider:
    """Клиент Gemini `generateContent` для генерации изображений.

    Ключ передаётся только в заголовке `x-goog-api-key`; reference-картинки
    уходят частями `inline_data` с base64. Транзиентные сбои (сеть, 429,
    5xx) повторяются через `tenacity` до `max_attempts` раз, постоянные
    (400/401/403/404/422 и прочие 4xx) падают сразу типизированной ошибкой.
    """

    def __init__(
        self,
        settings: StorySettings,
        client: httpx.AsyncClient | None = None,
        *,
        max_attempts: int = 3,
        retry_wait: Any | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._max_attempts = max_attempts
        self._retry_wait = (
            retry_wait
            if retry_wait is not None
            else wait_exponential(multiplier=0.5, min=0.5, max=8)
        )

    # --- Публичный API --------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        """Сгенерировать изображение, повторяя только транзиентные сбои."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientImageGenerationError),
            reraise=True,
        ):
            with attempt:
                return await self._request_and_parse(prompt, reference_images)

        # Недостижимо: AsyncRetrying с reraise=True либо вернёт результат,
        # либо поднимет исключение последней попытки.
        raise TransientImageGenerationError(  # pragma: no cover
            "Не удалось получить изображение от Gemini."
        )

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент, если он создавался здесь."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Внутренняя реализация -----------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.gemini_timeout
            )
        return self._client

    def _build_payload(
        self, prompt: str, reference_images: list[bytes] | None
    ) -> dict[str, Any]:
        """Собрать тело `generateContent` с текстом и reference-частями."""

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for reference in reference_images or []:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _REFERENCE_MIME_TYPE,
                        "data": base64.b64encode(reference).decode("ascii"),
                    }
                }
            )
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": _RESPONSE_MODALITIES},
        }

    async def _request_and_parse(
        self, prompt: str, reference_images: list[bytes] | None
    ) -> GeneratedImage:
        url = (
            f"{self._settings.gemini_base_url.rstrip('/')}"
            f"/v1beta/models/{self._settings.gemini_model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self._settings.gemini_api_key,
            "Content-Type": "application/json",
        }

        try:
            response = await self._build_client().post(
                url,
                json=self._build_payload(prompt, reference_images),
                headers=headers,
            )
        except httpx.TransportError as exc:
            raise TransientImageGenerationError(
                f"Сетевая ошибка при обращении к Gemini: {exc}"
            ) from exc

        if response.status_code in _PERMANENT_STATUS_CODES:
            raise PermanentImageGenerationError(
                "Gemini отклонил запрос "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientImageGenerationError(
                "Gemini временно недоступен "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code >= 400:
            raise PermanentImageGenerationError(
                f"Неожиданный ответ Gemini (HTTP {response.status_code})."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentImageGenerationError(
                "Gemini вернул ответ, который не удалось разобрать."
            ) from exc

        encoded = self._extract_encoded_image(data)
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PermanentImageGenerationError(
                "Gemini вернул некорректный base64-образ."
            ) from exc
        # Модель не поддерживает seed — не выдумываем воспроизводимость.
        return GeneratedImage(data=image_bytes, seed=None)

    @staticmethod
    def _extract_encoded_image(data: Any) -> str:
        """Достать base64-картинку из частей `candidates[0]`, оба написания."""

        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentImageGenerationError(
                "В ответе Gemini отсутствует изображение."
            ) from exc
        if not isinstance(parts, list):
            raise PermanentImageGenerationError(
                "В ответе Gemini отсутствует изображение."
            )

        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData")
            if not isinstance(inline, dict):
                inline = part.get("inline_data")
            if not isinstance(inline, dict):
                continue
            encoded = inline.get("data")
            if isinstance(encoded, str) and encoded:
                return encoded

        raise PermanentImageGenerationError(
            "В ответе Gemini отсутствует изображение."
        )


def make_image_provider(settings: StorySettings) -> ImageProvider:
    """Выбрать провайдера: mock при пустом ключе Gemini, иначе Gemini."""

    if not settings.gemini_api_key.strip():
        return MockImageProvider()
    return GeminiImageProvider(settings)


__all__ = [
    "GeneratedImage",
    "GeminiImageProvider",
    "ImageProvider",
    "MockImageProvider",
    "make_image_provider",
]
