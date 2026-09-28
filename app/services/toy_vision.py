"""Распознавание игрушки по фото: один вызов DeepSeek с изображением.

Сервис отвечает за продуктовое требование «текст и картинки описывают одного
и того же героя»: он единственный источник описания внешности игрушки
(`ToyDescription`), которое дальше попадает в тему планировщика, в подсказки
экстрактора и судьи и в промпт character sheet.

Живёт в `app/services/`, а не в пайплайне: внешний I/O проекта живёт в
сервисах, а пайплайн сказок остаётся переиспользуемым для сценариев без фото.
Ошибки сервиса — собственная таксономия `ToyVisionError`, чтобы граница бота
не зависела от ошибок внутреннего LLM-клиента пайплайна.

Модель распознавания — та же, что и для текста сказки (`deepseek_model`):
отдельная vision-модель в проекте не настроена, а описание игрушки — это та же
задача «спросить DeepSeek и получить строгий JSON», только с картинкой.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.utils.image_mime import UnsupportedImageTypeError, image_data_uri
from app.utils.logger import get_logger

logger = get_logger()

# Низкая температура: описание — источник правды для всех последующих
# промптов, и дрейфовать между прогонами оно не должно.
VISION_TEMPERATURE = 0.2

# Возможная markdown-обёртка вокруг JSON: ```json … ``` или ``` … ```.
_CODE_FENCE_RE = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```$", re.DOTALL
)

# HTTP-статусы, при которых повтор бессмыслен: аутентификация и валидация.
# Сюда попадает и отказ модели принимать изображения.
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})

# Задание модели: только по изображению, без домыслов, кратко.
TOY_VISION_PROMPT = (
    "Опиши игрушку с фотографии, строго на русском языке.\n"
    "Правила:\n"
    "1. Опирайся только на то, что видно на фотографии. Ничего не додумывай "
    "и не добавляй деталей, которых не видно.\n"
    "2. Если признак неразличим (например, материал), напиши «не видно».\n"
    "3. Не называй вымышленные имена, бренды и характеры — только вид.\n"
    "4. Отвечай кратко, одним JSON-объектом без пояснений и без markdown.\n\n"
    "Верни JSON ровно с такими ключами:\n"
    '{"kind": "вид игрушки (например, плюшевый медвежонок)",\n'
    ' "color": "цвет и узор",\n'
    ' "material": "материал",\n'
    ' "features": "2-4 характерные детали через запятую",\n'
    ' "short": "одно короткое предложение-описание (до 15 слов), пригодное '
    'как тема для сказки"}'
)


class ToyDescription(BaseModel):
    """Описание внешности игрушки — единый источник правды для промптов."""

    kind: str = Field(
        description="Вид игрушки: медвежонок, машинка, кукла и т. п."
    )
    color: str = Field(description="Цвет, узор и заметные цветовые детали.")
    material: str = Field(description="Материал, из которого сделана игрушка.")
    features: str = Field(
        description="Две-четыре характерные детали внешности через запятую."
    )
    short: str = Field(
        description="Короткое описание в одном предложении — тема для сказки."
    )


class ToyVisionError(Exception):
    """Базовая ошибка распознавания игрушки."""


class TransientToyVisionError(ToyVisionError):
    """Временная ошибка — запрос имеет смысл повторить один раз."""


class PermanentToyVisionError(ToyVisionError):
    """Постоянная ошибка — повтор не поможет, пользователю нужно объяснить."""


class ToyVisionService:
    """Клиент распознавания игрушки поверх DeepSeek Chat Completions."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        *,
        max_attempts: int = 2,
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

    async def describe(self, photo: bytes) -> ToyDescription:
        """Описать игрушку по байтам фотографии.

        `photo` — содержимое файла (JPEG, PNG или WebP); тип определяется по
        сигнатуре, а не по имени. Изображение уходит как data URI: без
        публичного URL и без промежуточного файла на диске.
        """

        data_uri = self._build_data_uri(photo)
        self._require_api_key()
        description = await self._request_with_retries(data_uri)
        logger.info(
            "Toy recognized: model={} kind={} short={}",
            self._settings.deepseek_model,
            description.kind,
            description.short,
        )
        return description

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент, если он создавался здесь."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Внутренняя реализация -----------------------------------------

    @staticmethod
    def _build_data_uri(photo: bytes) -> str:
        """Собрать data URI с MIME-типом, определённым по сигнатуре файла."""

        try:
            return image_data_uri(photo)
        except UnsupportedImageTypeError as exc:
            raise PermanentToyVisionError(
                "Не удалось определить формат фотографии игрушки. "
                "Пришлите фото в формате JPEG, PNG или WebP."
            ) from exc

    def _require_api_key(self) -> None:
        """Проверить ключ до сети: иначе 401 вводит в заблуждение."""

        if not self._settings.deepseek_api_key.strip():
            raise PermanentToyVisionError(
                "Не задан DEEPSEEK_API_KEY: распознавание игрушки "
                "не настроено, обращение к DeepSeek отменено."
            )

    def _build_payload(self, data_uri: str) -> dict[str, Any]:
        return {
            "model": self._settings.deepseek_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TOY_VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": VISION_TEMPERATURE,
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": self._settings.deepseek_max_tokens,
        }

    async def _request_with_retries(self, data_uri: str) -> ToyDescription:
        """Выполнить запрос, повторяя только временные сбои (один раз)."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientToyVisionError),
            reraise=True,
        ):
            with attempt:
                return await self._request_and_parse(data_uri)

        # Недостижимо: AsyncRetrying с reraise=True либо вернёт результат,
        # либо поднимет исключение последней попытки.
        raise TransientToyVisionError(  # pragma: no cover
            "Не удалось распознать игрушку по фотографии."
        )

    async def _request_and_parse(self, data_uri: str) -> ToyDescription:
        payload = self._build_payload(data_uri)
        data = await self._post_chat_completions(payload)
        content = self._extract_content(data)
        return self._parse(content)

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.deepseek_timeout
            )
        return self._client

    async def _post_chat_completions(self, payload: dict[str, Any]) -> Any:
        base = self._settings.deepseek_base_url.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._build_client().post(
                url, json=payload, headers=headers
            )
        except httpx.TransportError as exc:
            raise TransientToyVisionError(
                f"Сетевая ошибка при обращении к DeepSeek: {exc}"
            ) from exc

        if response.status_code in PERMANENT_STATUS_CODES:
            raise PermanentToyVisionError(
                "DeepSeek отклонил запрос с фотографией игрушки "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientToyVisionError(
                "DeepSeek временно недоступен "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code >= 400:
            raise PermanentToyVisionError(
                f"Неожиданный ответ DeepSeek (HTTP {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise PermanentToyVisionError(
                "DeepSeek вернул ответ, который не удалось разобрать."
            ) from exc

    @staticmethod
    def _extract_content(data: Any) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentToyVisionError(
                "В ответе DeepSeek отсутствует описание игрушки."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise PermanentToyVisionError(
                "DeepSeek вернул пустое описание игрушки."
            )
        return content

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        """Снять markdown-обёртку блока кода, как это делает LLM-клиент."""

        match = _CODE_FENCE_RE.match(content.strip())
        if match is None:
            return content.strip()
        return match.group("body").strip()

    @classmethod
    def _parse(cls, content: str) -> ToyDescription:
        """Разобрать описание по схеме, не подставляя выдуманных значений."""

        try:
            payload = json.loads(cls._strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise PermanentToyVisionError(
                f"DeepSeek вернул описание не в JSON: {exc}"
            ) from exc

        try:
            return ToyDescription.model_validate(payload)
        except ValidationError as exc:
            raise PermanentToyVisionError(
                "Описание игрушки от DeepSeek не соответствует схеме "
                f"{ToyDescription.__name__}: {exc}"
            ) from exc


__all__ = [
    "PERMANENT_STATUS_CODES",
    "TOY_VISION_PROMPT",
    "VISION_TEMPERATURE",
    "PermanentToyVisionError",
    "ToyDescription",
    "ToyVisionError",
    "ToyVisionService",
    "TransientToyVisionError",
]
