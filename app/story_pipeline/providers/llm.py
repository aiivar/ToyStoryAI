"""LLM-клиент пайплайна сказок: строгий JSON-обмен с DeepSeek.

Клиент скрывает вендор-специфику OpenAI-совместимого Chat Completions:
собирает запрос в JSON-режиме, разбирает ответ в pydantic-модель и
возвращает её вместе с `Usage`. Реализация за протоколом `LLMClient`,
чтобы этапы принимали фейк без сети. Транзиентные сбои повторяются
через `tenacity`, постоянные — падают сразу.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.story_pipeline.errors import (
    PermanentLLMError,
    SchemaValidationError,
    TransientLLMError,
)
from app.story_pipeline.settings import StorySettings

# Тип ответа LLM: конкретная pydantic-модель этапа.
T = TypeVar("T", bound=BaseModel)

# Возможная markdown-обёртка вокруг JSON: ```json … ``` или ``` … ```.
_CODE_FENCE_RE = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```$", re.DOTALL
)

# HTTP-статусы, при которых повтор бессмыслен: аутентификация и валидация.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


class Usage(BaseModel):
    """Расход токенов одного вызова LLM."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def render_schema(model: type[BaseModel]) -> str:
    """Отрендерить JSON-схему модели в текстовый блок для промпта.

    Схема берётся из самой pydantic-модели (`model_json_schema`), поэтому
    промпт этапа не может разойтись с моделью данных. Вывод детерминирован
    (`sort_keys`) и пригоден для вставки в пользовательский промпт как есть.
    """

    schema = model.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)


class LLMClient(Protocol):
    """Контракт текстовой модели: строгий JSON-ответ по pydantic-схеме."""

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
    ) -> tuple[T, Usage]:
        """Вернуть разобранную модель и расход токенов."""

        ...


class DeepSeekLLMClient:
    """Клиент DeepSeek поверх OpenAI-совместимого Chat Completions."""

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

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
    ) -> tuple[T, Usage]:
        """Запросить JSON по схеме и вернуть разобранную модель и Usage."""

        try:
            return await self._request_with_retries(
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
            )
        except SchemaValidationError as first_error:
            # Один корректирующий повтор с описанием ошибки: модель часто
            # исправляется, если явно указать, что не сошлось со схемой.
            try:
                return await self._request_with_retries(
                    system=system,
                    user=self._build_correction_prompt(user, first_error),
                    schema=schema,
                    temperature=temperature,
                )
            except SchemaValidationError as second_error:
                raise SchemaValidationError(
                    "LLM не вернул ответ по схеме даже после "
                    f"корректирующего повтора: {second_error}"
                ) from second_error

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

    async def _request_with_retries(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
    ) -> tuple[T, Usage]:
        """Выполнить запрос, повторяя только транзиентные сбои."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientLLMError),
            reraise=True,
        ):
            with attempt:
                return await self._request_and_parse(
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=temperature,
                )

        # Недостижимо: AsyncRetrying с reraise=True либо вернёт результат,
        # либо поднимет исключение последней попытки.
        raise TransientLLMError(  # pragma: no cover
            "Не удалось получить ответ DeepSeek."
        )

    async def _request_and_parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
    ) -> tuple[T, Usage]:
        payload = self._build_payload(
            system=system, user=user, temperature=temperature
        )
        data = await self._post_chat_completions(payload)
        content = self._extract_content(data)
        parsed = self._parse(schema=schema, content=content)
        return parsed, self._extract_usage(data)

    def _build_payload(
        self, *, system: str, user: str, temperature: float
    ) -> dict[str, Any]:
        return {
            "model": self._settings.deepseek_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": self._settings.deepseek_max_tokens,
        }

    async def _post_chat_completions(self, payload: dict[str, Any]) -> Any:
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
            raise TransientLLMError(
                f"Сетевая ошибка при обращении к DeepSeek: {exc}"
            ) from exc

        if response.status_code in _PERMANENT_STATUS_CODES:
            raise PermanentLLMError(
                "DeepSeek отклонил запрос "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientLLMError(
                "DeepSeek временно недоступен "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code >= 400:
            raise PermanentLLMError(
                f"Неожиданный ответ DeepSeek (HTTP {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise PermanentLLMError(
                "DeepSeek вернул ответ, который не удалось разобрать."
            ) from exc

    @staticmethod
    def _build_correction_prompt(user: str, error: SchemaValidationError) -> str:
        """Дополнить промпт описанием схемной ошибки для перезапроса."""

        return (
            f"{user}\n\n"
            "Предыдущий ответ не соответствовал требуемой JSON-схеме. "
            f"Ошибка: {error}. "
            "Верни исправленный ответ строго в формате JSON по той же схеме, "
            "без markdown-обёрток и пояснений."
        )

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        """Снять markdown-обёртку блока кода и вернуть полезную нагрузку."""

        match = _CODE_FENCE_RE.match(content.strip())
        if match is None:
            return content.strip()
        return match.group("body").strip()

    @classmethod
    def _parse(cls, schema: type[T], content: str) -> T:
        """Разобрать JSON-текст в модель, поднимая ошибку схемы при сбое."""

        try:
            payload = json.loads(cls._strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"LLM вернул не-JSON ответ: {exc}"
            ) from exc

        try:
            return cast(T, schema.model_validate(payload))
        except ValidationError as exc:
            raise SchemaValidationError(
                f"Ответ LLM не соответствует схеме {schema.__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_content(data: Any) -> str:
        """Достать текст ответа из конверта Chat Completions."""

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentLLMError(
                "В ответе DeepSeek отсутствует содержимое сообщения."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise PermanentLLMError("DeepSeek вернул пустой ответ.")
        return content

    @staticmethod
    def _extract_usage(data: Any) -> Usage:
        """Достать расход токенов; отсутствие usage допустимо."""

        usage_raw = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage_raw, dict):
            usage_raw = {}
        return Usage(
            prompt_tokens=usage_raw.get("prompt_tokens"),
            completion_tokens=usage_raw.get("completion_tokens"),
        )


__all__ = [
    "DeepSeekLLMClient",
    "LLMClient",
    "Usage",
    "render_schema",
]
