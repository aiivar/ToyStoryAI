"""Тесты сервиса генерации сказки (`app/services/text_generator.py`)."""

from __future__ import annotations

import json

import httpx
import pytest
from tenacity import wait_none

from app.config import Settings
from app.services.text_generator import (
    PermanentTextGenerationError,
    SYSTEM_PROMPT,
    TextGenerator,
    TransientTextGenerationError,
)

VALID_RESPONSE = {
    "choices": [{"message": {"content": "Жила-была храбрая игрушка..."}}]
}


def _make_generator(
    settings: Settings,
    handler,
) -> tuple[TextGenerator, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    generator = TextGenerator(settings, client, retry_wait=wait_none())
    return generator, client


# --- Задача 4.1: промпты и результат ----------------------------------------


async def test_prompt_contains_scenario_and_safety(
    settings: Settings,
) -> None:
    """Исходящий промпт содержит параметры сценария и требования безопасности."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    generator, client = _make_generator(settings, handler)
    try:
        text = await generator.generate(
            style="Волшебная",
            action="Полёт",
            toy_description="плюшевый заяц",
        )
    finally:
        await client.aclose()

    assert text == "Жила-была храбрая игрушка..."

    body = json.loads(captured[0].content.decode())
    system_content = body["messages"][0]["content"]
    user_content = body["messages"][1]["content"]

    assert body["messages"][0]["role"] == "system"
    assert "русск" in system_content
    assert "150" in system_content and "300" in system_content
    assert "3–10" in system_content or "3-10" in system_content
    assert "страшн" in system_content

    assert "плюшевый заяц" in user_content
    assert "Волшебная" in user_content
    assert "Полёт" in user_content


async def test_sends_bearer_authorization_header(
    settings: Settings,
) -> None:
    """Запрос аутентифицируется Bearer-заголовком, а не query-параметром."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    generator, client = _make_generator(settings, handler)
    try:
        await generator.generate(style="Смешная", action="Танец")
    finally:
        await client.aclose()

    request = captured[0]
    assert request.headers["Authorization"] == "Bearer "
    assert request.url.params.get("api_key") is None
    assert request.url.path.endswith("/chat/completions")


# --- Задача 4.2: модель из настроек -----------------------------------------


@pytest.mark.parametrize(
    "model",
    ["deepseek-flash", "deepseek-v4-pro"],
)
async def test_uses_configured_model(model: str) -> None:
    """В запрос отправляется модель, заданная в настройках."""

    configured = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
        deepseek_model=model,
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    generator, client = _make_generator(configured, handler)
    try:
        await generator.generate(style="Приключенческая", action="Путешествие")
    finally:
        await client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body["model"] == model


async def test_default_model_is_fast() -> None:
    """По умолчанию используется быстрая модель deepseek-flash."""

    config = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
    )
    assert config.deepseek_model == "deepseek-flash"


# --- Задача 4.3: повторы -----------------------------------------------------


async def test_transient_failure_is_retried_then_succeeds(
    settings: Settings,
) -> None:
    """Временный сбой повторяется, затем запрос завершается успехом."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=VALID_RESPONSE)

    generator, client = _make_generator(settings, handler)
    try:
        text = await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()

    assert calls["count"] == 2
    assert text == VALID_RESPONSE["choices"][0]["message"]["content"]


async def test_auth_error_is_not_retried(settings: Settings) -> None:
    """401 завершает шаг неудачей без исчерпания попыток."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    generator, client = _make_generator(settings, handler)
    try:
        with pytest.raises(PermanentTextGenerationError):
            await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()

    assert calls["count"] == 1


async def test_transient_errors_exhaust_attempts(settings: Settings) -> None:
    """После трёх неудачных попыток поднимается ошибка генерации."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "boom"})

    generator, client = _make_generator(settings, handler)
    try:
        with pytest.raises(TransientTextGenerationError):
            await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()

    assert calls["count"] == 3


async def test_malformed_response_raises_permanent(
    settings: Settings,
) -> None:
    """Ответ без текста сказки — нетранзиентная ошибка."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    generator, client = _make_generator(settings, handler)
    try:
        with pytest.raises(PermanentTextGenerationError):
            await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()


async def test_uses_max_tokens_from_settings(monkeypatch) -> None:
    """В запрос отправляется лимит токенов из настроек."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    configured = Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
        deepseek_max_tokens=4096,
    )
    generator, client = _make_generator(configured, handler)
    try:
        await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body["max_tokens"] == 4096


async def test_empty_content_with_length_hints_max_tokens(
    settings: Settings,
) -> None:
    """Пустой ответ с finish_reason=length подсказывает про лимит токенов."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": ""},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    generator, client = _make_generator(settings, handler)
    try:
        with pytest.raises(PermanentTextGenerationError) as exc:
            await generator.generate(style="Волшебная", action="Танец")
    finally:
        await client.aclose()

    assert "DEEPSEEK_MAX_TOKENS" in str(exc.value)


async def test_default_retry_wait_is_exponential(settings: Settings) -> None:
    """По умолчанию повторы используют экспоненциальную задержку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=VALID_RESPONSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    generator = TextGenerator(settings, client)
    try:
        assert type(generator._retry_wait).__name__ == "wait_exponential"
    finally:
        await client.aclose()


def test_system_prompt_is_russian_and_child_safe() -> None:
    """Системный промпт задаёт русский язык и детскую безопасность."""

    assert "русском" in SYSTEM_PROMPT
    assert "страшных" in SYSTEM_PROMPT
    assert "3–10" in SYSTEM_PROMPT
