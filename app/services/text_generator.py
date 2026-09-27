"""Генерация сказки через OpenAI-совместимый API DeepSeek.

Промпты фиксируют требования спеки: русский язык, 150–300 слов, игрушка —
главный герой, выбранные стиль и действие, без страшных сцен, для детей
3–10 лет. Транзиентные ошибки повторяются через tenacity.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger()

DEFAULT_TOY_DESCRIPTION = (
    "любимая игрушка ребёнка, изображённая на присланной фотографии"
)

SYSTEM_PROMPT = (
    "Ты — добрый детский сказочник. Пиши сказки только на русском языке. "
    "Объём сказки — от 150 до 300 слов. Главный герой — игрушка, которую "
    "описывает пользователь. Строго избегай страшных сцен, насилия, "
    "пугающих образов и любого контента, неподходящего детям 3–10 лет. "
    "История должна быть тёплой, безопасной и с хорошим концом."
)

# HTTP-статусы, при которых повтор бессмыслен: аутентификация и валидация.
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


class TextGenerationError(Exception):
    """Базовая ошибка генерации сказки."""


class TransientTextGenerationError(TextGenerationError):
    """Временная ошибка — запрос имеет смысл повторить."""


class PermanentTextGenerationError(TextGenerationError):
    """Нетранзиентная ошибка — повтор не поможет."""


class TextGenerator:
    """Клиент генерации сказки поверх DeepSeek Chat Completions."""

    def __init__(
        self,
        settings: Settings,
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
        *,
        style: str,
        action: str,
        toy_description: str = DEFAULT_TOY_DESCRIPTION,
    ) -> str:
        """Сгенерировать сказку для заданных стиля и действия."""

        payload: dict[str, Any] = {
            "model": self._settings.deepseek_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.build_user_prompt(
                        toy_description=toy_description,
                        style=style,
                        action=action,
                    ),
                },
            ],
            "temperature": 0.9,
            "max_tokens": self._settings.deepseek_max_tokens,
            "stream": False,
        }

        return await self._request_with_retries(payload)

    @staticmethod
    def build_user_prompt(
        *,
        toy_description: str,
        style: str,
        action: str,
    ) -> str:
        """Собрать пользовательский промпт из параметров сценария."""

        return (
            f"Игрушка: {toy_description}.\n"
            f"Стиль сказки: {style}.\n"
            f"Действие игрушки: {action}.\n\n"
            "Напиши сказку на русском языке, где эта игрушка — главный герой "
            f"и выполняет действие «{action}». Соблюдай стиль «{style}». "
            "Объём — от 150 до 300 слов. Не добавляй страшные сцены и "
            "контент, неподходящий детям 3–10 лет."
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
                timeout=self._settings.deepseek_timeout
            )
        return self._client

    async def _request_with_retries(self, payload: dict[str, Any]) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientTextGenerationError),
            reraise=True,
        ):
            with attempt:
                return await self._post_chat_completions(payload)

        # Недостижимо: AsyncRetrying с reraise=True поднимает исключение,
        # либо возвращает результат на успешной попытке.
        raise PermanentTextGenerationError(  # pragma: no cover
            "Не удалось сгенерировать сказку."
        )

    async def _post_chat_completions(self, payload: dict[str, Any]) -> str:
        url = f"{self._settings.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._build_client().post(
                url, json=payload, headers=headers
            )
        except httpx.TransportError as exc:
            raise TransientTextGenerationError(
                f"Сетевая ошибка при обращении к DeepSeek: {exc}"
            ) from exc

        if response.status_code in PERMANENT_STATUS_CODES:
            raise PermanentTextGenerationError(
                "DeepSeek отклонил запрос сказки "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientTextGenerationError(
                "DeepSeek временно недоступен "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code >= 400:
            raise PermanentTextGenerationError(
                f"Неожиданный ответ DeepSeek (HTTP {response.status_code})."
            )

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentTextGenerationError(
                "DeepSeek вернул ответ, который не удалось разобрать."
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentTextGenerationError(
                "В ответе DeepSeek отсутствует текст сказки."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            finish_reason = data["choices"][0].get("finish_reason")
            hint = (
                " Похоже, лимит токенов израсходован на размышления — "
                "увеличьте DEEPSEEK_MAX_TOKENS."
                if finish_reason == "length"
                else ""
            )
            raise PermanentTextGenerationError(
                f"DeepSeek вернул пустую сказку (finish_reason={finish_reason})."
                f"{hint}"
            )

        return content.strip()


__all__ = [
    "DEFAULT_TOY_DESCRIPTION",
    "PermanentTextGenerationError",
    "SYSTEM_PROMPT",
    "TextGenerationError",
    "TextGenerator",
    "TransientTextGenerationError",
]
