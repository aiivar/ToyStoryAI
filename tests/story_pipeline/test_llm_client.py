"""Тесты LLM-клиента пайплайна (`app/story_pipeline/providers/llm.py`).

Публичная граница клиента — HTTP, поэтому внешний API подменяется
`httpx.MockTransport`. Внутренние методы клиента не мокаются: тесты
переживают любой рефакторинг его устройства.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from tenacity import wait_none

from app.story_pipeline.errors import (
    PermanentLLMError,
    SchemaValidationError,
    TransientLLMError,
)
from app.story_pipeline.providers.llm import DeepSeekLLMClient, render_schema
from app.story_pipeline.settings import StorySettings


class _Dummy(BaseModel):
    """Мини-модель для проверки разбора ответа."""

    name: str
    count: int


def _chat_response(
    content: str,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 22,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Собрать ответ DeepSeek в формате Chat Completions."""

    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


Handler = Callable[[httpx.Request], httpx.Response]


def _make_client(
    handler: Handler,
) -> tuple[DeepSeekLLMClient, httpx.AsyncClient]:
    """Создать клиент с подменённым транспортом и без задержек ретраев."""

    settings = StorySettings(
        _env_file=None,  # type: ignore[call-arg]
        deepseek_api_key="test-key",
        deepseek_model="deepseek-test",
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DeepSeekLLMClient(settings, http_client, retry_wait=wait_none())
    return client, http_client


async def test_parses_valid_json_response_with_usage() -> None:
    """Корректный JSON-ответ разбирается в модель и Usage с токенами."""

    content = json.dumps({"name": "заяц", "count": 3})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(content))

    client, http_client = _make_client(handler)
    try:
        model, usage = await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.7,
        )
    finally:
        await http_client.aclose()

    assert isinstance(model, _Dummy)
    assert model.name == "заяц"
    assert model.count == 3
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 22


async def test_sends_openai_compatible_json_request() -> None:
    """Запрос идёт на chat/completions с Bearer-ключом и JSON-режимом."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        content = json.dumps({"name": "заяц", "count": 1})
        return httpx.Response(200, json=_chat_response(content))

    client, http_client = _make_client(handler)
    try:
        await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.3,
        )
    finally:
        await http_client.aclose()

    request = captured[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Content-Type"] == "application/json"

    body = json.loads(request.content.decode())
    assert body["model"] == "deepseek-test"
    assert body["messages"] == [
        {"role": "system", "content": "система"},
        {"role": "user", "content": "пользователь"},
    ]
    assert body["temperature"] == 0.3
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["max_tokens"] == 8192


@pytest.mark.parametrize("fence_open", ["```json", "```"])
async def test_strips_markdown_code_fence(fence_open: str) -> None:
    """Ответ, обёрнутый в блок кода, распознаётся после снятия забора."""

    payload = json.dumps({"name": "лис", "count": 2})
    content = f"{fence_open}\n{payload}\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(content))

    client, http_client = _make_client(handler)
    try:
        model, _ = await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.5,
        )
    finally:
        await http_client.aclose()

    assert model.name == "лис"
    assert model.count == 2


async def test_broken_json_raises_schema_validation_error() -> None:
    """НеJSON-ответ поднимает типизированную ошибку схемы, а не ValueError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response("это не JSON"))

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(SchemaValidationError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()


async def test_transient_error_is_retried_then_succeeds() -> None:
    """503 повторяется, и вторая успешная попытка возвращает результат."""

    calls = {"count": 0}
    content = json.dumps({"name": "медведь", "count": 7})

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_chat_response(content))

    client, http_client = _make_client(handler)
    try:
        model, _ = await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.7,
        )
    finally:
        await http_client.aclose()

    assert calls["count"] == 2
    assert model.name == "медведь"


async def test_transient_errors_exhaust_attempts() -> None:
    """Исчерпав попытки, клиент поднимает транзиентную ошибку."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(TransientLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


@pytest.mark.parametrize("api_key", ["", "   "])
async def test_empty_api_key_fails_before_http_call(api_key: str) -> None:
    """Пустой ключ — постоянная ошибка конфигурации до любого HTTP-запроса."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_chat_response("{}"))

    settings = StorySettings(
        _env_file=None,  # type: ignore[call-arg]
        deepseek_api_key=api_key,
        deepseek_model="deepseek-test",
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DeepSeekLLMClient(settings, http_client, retry_wait=wait_none())
    try:
        with pytest.raises(PermanentLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    # Ни одного сетевого вызова: ошибка конфигурации видна сразу.
    assert calls["count"] == 0


async def test_permanent_auth_error_is_not_retried() -> None:
    """401 падает сразу постоянной ошибкой, не исчерпывая попытки."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(PermanentLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


async def test_schema_mismatch_triggers_one_correction() -> None:
    """Ответ, не бьющийся со схемой, исправляется одним перезапросом."""

    captured: list[httpx.Request] = []
    mismatched = json.dumps({"name": "лис"})  # нет обязательного count
    valid = json.dumps({"name": "лис", "count": 5})

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if len(captured) == 1:
            return httpx.Response(200, json=_chat_response(mismatched))
        return httpx.Response(200, json=_chat_response(valid))

    client, http_client = _make_client(handler)
    try:
        model, _ = await client.complete_json(
            system="система",
            user="исходный запрос",
            schema=_Dummy,
            temperature=0.7,
        )
    finally:
        await http_client.aclose()

    assert model.count == 5
    assert len(captured) == 2

    correction_body = json.loads(captured[1].content.decode())
    assert correction_body["messages"][0]["content"] == "система"
    correction_user = correction_body["messages"][1]["content"]
    assert "исходный запрос" in correction_user
    assert correction_user != "исходный запрос"
    # В корректирующем повторе передано описание схемной ошибки.
    assert "count" in correction_user


async def test_schema_mismatch_twice_raises_schema_validation_error() -> None:
    """Если корректирующий повтор не помог — поднимается ошибка схемы."""

    calls = {"count": 0}
    mismatched = json.dumps({"name": "лис"})

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_chat_response(mismatched))

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(SchemaValidationError):
            await client.complete_json(
                system="система",
                user="исходный запрос",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 2


async def test_truncated_response_reports_token_limit_not_schema() -> None:
    """Обрыв по лимиту токенов называет лимит, а не «невалидную схему».

    Обрезанный по `max_tokens` ответ почти всегда невалиден, и прежняя
    формулировка сваливала бы вину на модель, хотя виноват лимит, и
    подсказывала не то, что нужно увеличить.
    """

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json=_chat_response(
                json.dumps({"name": "заяц"}),
                finish_reason="length",
            ),
        )

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(PermanentLLMError) as failure:
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    message = str(failure.value)
    assert "DEEPSEEK_MAX_TOKENS" in message
    assert "не успел дописать" in message
    # Повтор по схеме не тратится на заведомо обрезанный ответ: второй запрос
    # упёрся бы в тот же лимит.
    assert calls["count"] == 1


async def test_truncated_response_is_not_retried_as_schema_error() -> None:
    """Обрыв отличается от схемной ошибки и не уходит в корректирующий повтор."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json=_chat_response(
                json.dumps({"name": "заяц"}),
                finish_reason="length",
            ),
        )

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(PermanentLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


async def test_complete_invalid_response_still_uses_correction() -> None:
    """Полный невалидный ответ по-прежнему чинится корректирующим повтором.

    Регрессия на разведение двух видов отказа: постоянная ошибка обрыва не
    должна была отнять у клиента его обычную схемную починку.
    """

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                200,
                json=_chat_response(
                    json.dumps({"name": "заяц"}), finish_reason="length"
                ),
            )
        return httpx.Response(
            200,
            json=_chat_response(json.dumps({"name": "заяц", "count": 4})),
        )

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(PermanentLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


async def test_full_schema_mismatch_still_uses_one_correction() -> None:
    """`finish_reason == "stop"` с невалидным ответом — обычная схемная ошибка."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json=_chat_response(json.dumps({"name": "заяц"})),
        )

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(SchemaValidationError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 2


async def test_request_carries_default_max_tokens_of_8192() -> None:
    """Дефолт лимита поднят вдвое: 12 страниц длинного текена в 4096 не влезают.

    Проверяется фактическое тело запроса, а не значение настройки: дефолт,
    который не доезжает до вендора, не спасает ни один живой прогон.
    """

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json=_chat_response(json.dumps({"name": "заяц", "count": 1})),
        )

    client, http_client = _make_client(handler)
    try:
        await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.7,
        )
    finally:
        await http_client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body["max_tokens"] == 8192


async def test_network_error_is_transient_and_retried() -> None:
    """Сетевая ошибка классифицируется как транзиентная и повторяется."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("нет сети", request=request)

    client, http_client = _make_client(handler)
    try:
        with pytest.raises(TransientLLMError):
            await client.complete_json(
                system="система",
                user="пользователь",
                schema=_Dummy,
                temperature=0.7,
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


async def test_missing_usage_yields_empty_tokens() -> None:
    """Отсутствие блока usage не роняет разбор: токены остаются None."""

    content = json.dumps({"name": "ёж", "count": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    client, http_client = _make_client(handler)
    try:
        _, usage = await client.complete_json(
            system="система",
            user="пользователь",
            schema=_Dummy,
            temperature=0.7,
        )
    finally:
        await http_client.aclose()

    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None


def test_render_schema_contains_required_fields_and_is_deterministic() -> None:
    """Схема модели рендерится с обязательными полями и без разброса."""

    rendered = render_schema(_Dummy)

    assert isinstance(rendered, str)
    assert rendered.strip()
    assert "name" in rendered
    assert "count" in rendered
    assert "required" in rendered
    assert render_schema(_Dummy) == rendered
