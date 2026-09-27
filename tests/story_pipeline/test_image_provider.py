"""Тесты провайдеров изображений (`app/story_pipeline/providers/image.py`).

Публичная граница Gemini-провайдера — HTTP, поэтому внешний API
подменяется `httpx.MockTransport`; внутренние методы не мокаются. Mock
провайдер проверяется на детерминизм и на отсутствие сети.
"""

from __future__ import annotations

import base64
import json

from collections.abc import Callable

import httpx
import pytest
from tenacity import wait_none

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    TransientImageGenerationError,
)
from app.story_pipeline.providers.image import (
    GeminiImageProvider,
    MockImageProvider,
    make_image_provider,
)
from app.story_pipeline.settings import StorySettings

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _settings(**overrides: object) -> StorySettings:
    """Настройки пайплайна без чтения `.env`, с переопределениями."""

    values: dict[str, object] = {"_env_file": None}
    values.update(overrides)
    return StorySettings(**values)  # type: ignore[arg-type]


async def test_mock_provider_is_deterministic_for_same_seed() -> None:
    """Один и тот же seed даёт одинаковые байты, разные — разные."""

    provider = MockImageProvider()

    first = await provider.generate("промпт", seed=42)
    second = await provider.generate("промпт", seed=42)
    other = await provider.generate("промпт", seed=43)

    assert first.data == second.data
    assert first.data != other.data


async def test_mock_provider_returns_valid_png() -> None:
    """Mock отдаёт валидный PNG и прокидывает seed без изменений."""

    provider = MockImageProvider()

    image = await provider.generate("промпт", seed=7)

    assert image.data.startswith(PNG_SIGNATURE)
    assert image.seed == 7


async def test_mock_provider_has_no_http_client() -> None:
    """Mock не держит HTTP-клиент и работает офлайн."""

    provider = MockImageProvider()

    assert not hasattr(provider, "_client")

    image = await provider.generate("промпт", seed=1)
    assert image.data.startswith(PNG_SIGNATURE)


def test_factory_without_key_returns_mock() -> None:
    """Пустой ключ Gemini включает mock-провайдер."""

    provider = make_image_provider(_settings(gemini_api_key=""))

    assert isinstance(provider, MockImageProvider)


def test_factory_with_key_returns_gemini() -> None:
    """Непустой ключ Gemini включает реальный провайдер."""

    provider = make_image_provider(_settings(gemini_api_key="secret"))

    assert isinstance(provider, GeminiImageProvider)


# --- Gemini: успешный вызов --------------------------------------------


def _gemini_image_response(
    image_bytes: bytes, *, key: str = "inlineData"
) -> dict[str, object]:
    """Собрать ответ Gemini с base64-изображением в указанном написании."""

    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "готово"},
                        {
                            key: {
                                "mimeType": "image/png",
                                "data": base64.b64encode(image_bytes).decode(
                                    "ascii"
                                ),
                            }
                        },
                    ]
                }
            }
        ]
    }


def _make_gemini(
    handler: "Callable[[httpx.Request], httpx.Response]",
) -> tuple[GeminiImageProvider, httpx.AsyncClient]:
    """Создать провайдер с подменённым транспортом и без задержек ретраев."""

    settings = _settings(
        gemini_api_key="test-key",
        gemini_model="gemini-test-image",
        gemini_base_url="https://gemini.test",
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiImageProvider(
        settings, http_client, retry_wait=wait_none()
    )
    return provider, http_client


async def test_gemini_success_sends_reference_parts_and_parses_image() -> None:
    """Успешный вызов шлёт ключ в заголовке и reference-часть, парсит ответ."""

    captured: list[httpx.Request] = []
    reference = b"\x89PNG-reference-bytes"
    expected = b"\x89PNG-generated-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_gemini_image_response(expected))

    provider, http_client = _make_gemini(handler)
    try:
        image = await provider.generate(
            "нарисуй зайца",
            reference_images=[reference],
        )
    finally:
        await http_client.aclose()

    assert image.data == expected
    assert image.seed is None

    request = captured[0]
    assert str(request.url) == (
        "https://gemini.test/v1beta/models/gemini-test-image:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(request.url)

    body = json.loads(request.content.decode())
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "нарисуй зайца"}
    assert parts[1] == {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(reference).decode("ascii"),
        }
    }
    assert body["generationConfig"] == {"responseModalities": ["IMAGE"]}


async def test_gemini_parses_inline_data_snake_case_spelling() -> None:
    """Ответ с ключом `inline_data` тоже разбирается."""

    expected = b"\x89PNG-snake-case"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_gemini_image_response(expected, key="inline_data")
        )

    provider, http_client = _make_gemini(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.data == expected


async def test_gemini_without_references_sends_only_text() -> None:
    """Без reference-картинок тело содержит только текстовую часть."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_gemini_image_response(b"png"))

    provider, http_client = _make_gemini(handler)
    try:
        await provider.generate("только текст")
    finally:
        await http_client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body["contents"][0]["parts"] == [{"text": "только текст"}]


async def test_gemini_transient_error_is_retried_then_succeeds() -> None:
    """503 повторяется, и вторая успешная попытка возвращает картинку."""

    calls = {"count": 0}
    expected = b"\x89PNG-after-retry"

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_gemini_image_response(expected))

    provider, http_client = _make_gemini(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.data == expected
    assert calls["count"] == 2


async def test_gemini_transient_errors_exhaust_attempts() -> None:
    """Исчерпав попытки, провайдер поднимает транзиентную ошибку."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "boom"})

    provider, http_client = _make_gemini(handler)
    try:
        with pytest.raises(TransientImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


async def test_gemini_network_error_is_transient_and_retried() -> None:
    """Сетевая ошибка классифицируется как транзиентная и повторяется."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("нет сети", request=request)

    provider, http_client = _make_gemini(handler)
    try:
        with pytest.raises(TransientImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


async def test_gemini_permanent_auth_error_is_not_retried() -> None:
    """401 падает сразу постоянной ошибкой, ровно один запрос."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    provider, http_client = _make_gemini(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


@pytest.mark.parametrize("status", [400, 403, 404, 422])
async def test_gemini_permanent_client_errors_are_not_retried(
    status: int,
) -> None:
    """Прочие постоянные 4xx падают сразу, без исчерпания попыток."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(status, json={"error": "nope"})

    provider, http_client = _make_gemini(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"candidates": []},
        {"candidates": [{"content": {"parts": [{"text": "нет картинки"}]}}]},
        {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"data": ""}}]}}
            ]
        },
    ],
)
async def test_gemini_missing_image_raises_permanent_error(
    payload: dict[str, object],
) -> None:
    """Ответ без изображения поднимает типизированную постоянную ошибку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider, http_client = _make_gemini(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()
