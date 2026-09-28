"""Провайдеры генерации изображений: RouterAI, Gemini и детерминированный mock.

Контракт скрыт за протоколом `ImageProvider`, чтобы этап иллюстрирования
принимал фейк без сети. `RouterAIImageProvider` — рабочий провайдер: один
POST на `/images`, ключ только в заголовке, reference-картинки как data URI с
типом по сигнатуре файла, разбор `data[0].b64_json` вместе с типом файла и
стоимостью вызова. `GeminiImageProvider` изолирует вендор-специфику
`generateContent`; `MockImageProvider` строит валидный PNG из seed
стандартной библиотекой и включается при пустом ключе.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
import zlib
from typing import Any, Protocol

import httpx
from loguru import logger
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
from app.story_pipeline.settings import (
    SUPPORTED_ASPECT_RATIOS,
    StorySettings,
)
from app.utils.image_mime import (
    UnsupportedImageTypeError,
    detect_image_mime_type,
    image_data_uri,
)

# Сигнатура PNG-файла: первые восемь байт любого PNG.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Сторона квадратного mock-изображения в пикселях. Небольшой размер
# достаточен, чтобы файл был валидным PNG, и не раздувает тесты и вывод.
_MOCK_IMAGE_SIZE = 32

# Тип цвета PNG: truecolor (по три байта на пиксель, без альфа-канала).
_PNG_COLOR_TYPE_RGB = 2

# Глубина цвета PNG в битах на канал.
_PNG_BIT_DEPTH = 8

# MIME reference-картинок Gemini: character sheet и страницы сохраняются
# как PNG, а провайдер не сообщает тип в ответе.
_REFERENCE_MIME_TYPE = "image/png"

# Запрошенные модальности ответа Gemini: только изображение.
_RESPONSE_MODALITIES = ["IMAGE"]

# HTTP-статусы, при которых повтор бессмыслен: аутентификация и
# валидация запроса.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})

# Путь генерации изображений RouterAI относительно базового URL.
_IMAGES_PATH = "/images"

# Тип файла по умолчанию, если провайдер его не сообщил: mock и Gemini
# отдают PNG, поэтому предположение проверяемо и не ломает mock-прогон.
_DEFAULT_MEDIA_TYPE = "image/png"


class GeneratedImage(BaseModel):
    """Результат генерации: байты, seed, тип файла и стоимость вызова.

    Тип и стоимость приходят от провайдера, поэтому имя файла и оценка
    прогона по деньгам не зависят от догадок. Оба поля — числа и строки,
    то есть сериализуются в `story.json` вместе с остальным состоянием и
    не тащат в него бинари.
    """

    data: bytes
    seed: int | None = None
    media_type: str = _DEFAULT_MEDIA_TYPE
    cost: float | None = None


class ImageProvider(Protocol):
    """Контракт генератора изображений: промпт → байты картинки."""

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Сгенерировать изображение по промпту и reference-картинкам.

        `aspect_ratio` переопределяет значение `STORY_IMAGE_ASPECT_RATIO` для
        одного вызова: страницы рисуются в книжной пропорции, а
        character sheet — в квадратной.
        """

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
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Вернуть PNG, выведенный из seed; промпт и референсы не нужны."""

        effective_seed = 0 if seed is None else seed
        return GeneratedImage(data=_mock_png(effective_seed), seed=seed)


class RouterAIImageProvider:
    """Клиент RouterAI `/images`: один запрос — одна картинка.

    Ключ передаётся только в заголовке `Authorization`: ни в URL, ни в теле
    запроса его нет. Reference-картинки уходят как data URI, тип в URI
    определяется по сигнатуре байтов. В ответе ожидается
    `data[0].b64_json`, тип файла — в `data[0].media_type`, стоимость
    вызова — в `usage.cost`.
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
        # Сумма по прогону накапливается здесь: стоимость каждой картинки
        # приходит из `usage.cost`, а итог нужен, чтобы оценить запуск.
        self._total_cost: float | None = None

    # --- Публичный API --------------------------------------------------

    @property
    def total_cost(self) -> float | None:
        """Суммарная стоимость вызовов провайдера; `None` — неизвестно."""

        return self._total_cost

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Сгенерировать изображение, повторяя транзиентные сбои."""

        self._require_api_key()
        aspect_ratio = self._resolve_aspect_ratio(aspect_ratio)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientImageGenerationError),
            reraise=True,
        ):
            with attempt:
                return await self._request_and_parse(
                    prompt, seed, reference_images, aspect_ratio
                )

        # Недостижимо: AsyncRetrying с reraise=True либо вернёт результат,
        # либо поднимет исключение последней попытки.
        raise TransientImageGenerationError(  # pragma: no cover
            "Не удалось получить изображение от RouterAI."
        )

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент, если он создавался здесь."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Внутренняя реализация -----------------------------------------

    def _require_api_key(self) -> None:
        """Проверить ключ до обращения к сети, подняв постоянную ошибку."""

        if not self._settings.routerai_api_key.strip():
            raise PermanentImageGenerationError(
                "Не задан ROUTERAI_API_KEY: генератор изображений не "
                "настроен, обращение к RouterAI отменено."
            )

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.routerai_timeout
            )
        return self._client

    def _resolve_aspect_ratio(self, aspect_ratio: str | None) -> str | None:
        """Вернуть соотношение сторон вызова: аргумент важнее настройки.

        Значение вне списка провайдера отклоняется до сетевого вызова: платный
        запрос с заведомо отвергаемым полем — чистый расход. Пустое значение
        означает «не отправлять соотношение» и оставляет умолчание провайдера.
        """

        value = (
            self._settings.story_image_aspect_ratio
            if aspect_ratio is None
            else aspect_ratio
        )
        stripped = value.strip()
        if not stripped:
            return None
        if stripped not in SUPPORTED_ASPECT_RATIOS:
            supported = ", ".join(SUPPORTED_ASPECT_RATIOS)
            raise PermanentImageGenerationError(
                f"Неподдерживаемое соотношение сторон {stripped!r}: "
                f"допустимые значения — {supported}."
            )
        return stripped

    def _build_payload(
        self,
        prompt: str,
        seed: int | None,
        reference_images: list[bytes] | None,
        aspect_ratio: str | None,
    ) -> dict[str, Any]:
        """Собрать тело `/images`: модель, промпт и необязательные поля."""

        payload: dict[str, Any] = {
            "model": self._settings.routerai_image_model,
            "prompt": prompt,
            "n": 1,
        }
        if seed is not None:
            payload["seed"] = seed
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        references = self._build_references(reference_images)
        if references:
            payload["input_references"] = references
        return payload

    @staticmethod
    def _build_references(
        reference_images: list[bytes] | None,
    ) -> list[dict[str, Any]]:
        """Превратить байты референсов в ссылки data URI для тела запроса."""

        references: list[dict[str, Any]] = []
        for reference in reference_images or []:
            try:
                uri = image_data_uri(reference)
            except UnsupportedImageTypeError as exc:
                # Референс приходит из нашего же провайдера или из фото,
                # проверенного приёмом: повтор не поможет.
                raise PermanentImageGenerationError(
                    "Не удалось определить формат reference-изображения "
                    "по первым байтам: ожидается JPEG, PNG или WebP."
                ) from exc
            references.append(
                {"type": "image_url", "image_url": {"url": uri}}
            )
        return references

    async def _request_and_parse(
        self,
        prompt: str,
        seed: int | None,
        reference_images: list[bytes] | None,
        aspect_ratio: str | None,
    ) -> GeneratedImage:
        url = f"{self._settings.routerai_base_url.rstrip('/')}{_IMAGES_PATH}"
        headers = {
            "Authorization": f"Bearer {self._settings.routerai_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._build_client().post(
                url,
                json=self._build_payload(
                    prompt, seed, reference_images, aspect_ratio
                ),
                headers=headers,
            )
        except httpx.TransportError as exc:
            raise TransientImageGenerationError(
                f"Сетевая ошибка при обращении к RouterAI: {exc}"
            ) from exc

        if response.status_code in _PERMANENT_STATUS_CODES:
            raise PermanentImageGenerationError(
                "RouterAI отклонил запрос "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientImageGenerationError(
                "RouterAI временно недоступен "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code >= 400:
            raise PermanentImageGenerationError(
                f"Неожиданный ответ RouterAI (HTTP {response.status_code})."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentImageGenerationError(
                "RouterAI вернул ответ, который не удалось разобрать."
            ) from exc

        encoded, media_type, cost = self._extract_image(data)
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PermanentImageGenerationError(
                "RouterAI вернул некорректный base64-образ."
            ) from exc
        image = GeneratedImage(
            data=image_bytes,
            seed=seed,
            media_type=_resolve_media_type(media_type, image_bytes),
            cost=cost,
        )
        self._record_cost(cost, image)
        return image

    def _record_cost(self, cost: float | None, image: GeneratedImage) -> None:
        """Прибавить стоимость вызова к сумме и записать её в лог.

        В лог уходят только числа и тип файла: ни промпт, ни ключ, ни тело
        запроса. Стоимость приходит в рублях от провайдера, поэтому печатается
        как есть, без округления и без пересчёта в другую валюту.
        """

        if cost is not None:
            self._total_cost = (self._total_cost or 0.0) + cost
        total = (
            f"{self._total_cost:.3f}"
            if self._total_cost is not None
            else "unknown"
        )
        logger.info(
            "routerai: image generated bytes={} media_type={} cost={} "
            "total_cost={}",
            len(image.data),
            image.media_type,
            f"{cost:.3f}" if cost is not None else "unknown",
            total,
        )

    @staticmethod
    def _extract_image(data: Any) -> tuple[str, str | None, float | None]:
        """Достать base64-картинку, тип файла и стоимость из ответа."""

        try:
            item = data["data"][0]
            encoded = item["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentImageGenerationError(
                "В ответе RouterAI отсутствует изображение."
            ) from exc
        if not isinstance(encoded, str) or not encoded:
            raise PermanentImageGenerationError(
                "В ответе RouterAI отсутствует изображение."
            )

        media_type = item.get("media_type") if isinstance(item, dict) else None
        if not isinstance(media_type, str):
            media_type = None
        return encoded, media_type, _extract_cost(data)


def _resolve_media_type(
    media_type: str | None, image_bytes: bytes
) -> str:
    """Определить тип файла: ответ провайдера, иначе сигнатура байтов.

    Тип из ответа приоритетен — это то, чем вендор собирается сохранить
    картинку. Значение без префикса вида (`jpeg`) приводится к полному
    MIME-виду, чтобы дальше работать с единым `image/...` форматом. Если тип
    не пришёл, он определяется по содержимому: конвертация запрещена, а
    расширение файла и вложения строятся именно по этому типу.
    """

    if media_type and media_type.strip():
        value = media_type.strip().lower()
        return value if "/" in value else f"image/{value}"
    return detect_image_mime_type(image_bytes) or _DEFAULT_MEDIA_TYPE


def _extract_cost(data: Any) -> float | None:
    """Достать стоимость вызова из `usage.cost`; отсутствие — `None`.

    Значение приходит в рублях; если поля нет или оно не число, стоимость
    остаётся неизвестной — выдумывать её нельзя.
    """

    usage = data.get("usage") if isinstance(data, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    return float(cost)


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
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Сгенерировать изображение, повторяя только транзиентные сбои.

        `aspect_ratio` и `seed` игнорируются: `generateContent` не принимает
        ни того, ни другого, поэтому результат остаётся честно
        невоспроизводимым.
        """

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
    """Выбрать провайдера: mock при пустом ключе RouterAI, иначе RouterAI."""

    if not settings.routerai_api_key.strip():
        return MockImageProvider()
    return RouterAIImageProvider(settings)


__all__ = [
    "GeneratedImage",
    "GeminiImageProvider",
    "ImageProvider",
    "MockImageProvider",
    "RouterAIImageProvider",
    "make_image_provider",
]
