"""Тесты `RouterAIImageProvider` (`app/story_pipeline/providers/image.py`).

Публичная граница провайдера — HTTP, поэтому внешний API подменяется
`httpx.MockTransport`: тесты проверяют реальный запрос (URL, заголовки, тело)
и реальный разбор ответа, а не внутренние методы. Ответы повторяют форму,
подтверждённую живым вызовом `/images`: `data[0].b64_json`, `media_type` и
`usage.cost`.
"""

from __future__ import annotations

import base64
import json

from collections.abc import Callable

import httpx
import pytest
from loguru import logger
from tenacity import wait_none

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    TransientImageGenerationError,
)
from app.story_pipeline.providers.image import (
    GeneratedImage,
    RouterAIImageProvider,
)
from app.story_pipeline.settings import StorySettings

# Подписи файлов: референс приходит от MAX транскодированным в WebP.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00"

API_KEY = "routerai-test-key"
BASE_URL = "https://routerai.test/api/v1"
IMAGE_PATH = f"{BASE_URL}/images"


def _settings(**overrides: object) -> StorySettings:
    """Настройки пайплайна без чтения `.env`, с переопределениями."""

    values: dict[str, object] = {
        "_env_file": None,
        "routerai_api_key": API_KEY,
        "routerai_image_model": "google/routerai-test-image",
        "routerai_base_url": BASE_URL,
        "story_image_aspect_ratio": "4:3",
    }
    values.update(overrides)
    return StorySettings(**values)  # type: ignore[arg-type]


def _routerai_response(
    image_bytes: bytes,
    *,
    media_type: str | None = "image/jpeg",
    cost: float | None = 3.756,
) -> dict[str, object]:
    """Собрать ответ RouterAI `/images` в подтверждённой живьём форме."""

    item: dict[str, object] = {
        "b64_json": base64.b64encode(image_bytes).decode("ascii")
    }
    if media_type is not None:
        item["media_type"] = media_type
    usage: dict[str, object] = {"total_tokens": 1120}
    if cost is not None:
        usage["cost"] = cost
    return {"created": 1759000000, "data": [item], "usage": usage}


def _make_routerai(
    handler: "Callable[[httpx.Request], httpx.Response]",
    **overrides: object,
) -> tuple[RouterAIImageProvider, httpx.AsyncClient]:
    """Создать провайдер с подменённым транспортом и без задержек ретраев."""

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = RouterAIImageProvider(
        _settings(**overrides), http_client, retry_wait=wait_none()
    )
    return provider, http_client


# --- Успешный вызов ----------------------------------------------------


async def test_success_sends_documented_request_and_parses_image() -> None:
    """Запрос соответствует документации, ответ разбирается в картинку."""

    captured: list[httpx.Request] = []
    reference = WEBP_BYTES + b"reference-payload"
    expected = JPEG_BYTES + b"generated-payload"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, json=_routerai_response(expected, cost=3.756)
        )

    provider, http_client = _make_routerai(handler)
    try:
        image = await provider.generate(
            "нарисуй зайца", seed=1001, reference_images=[reference]
        )
    finally:
        await http_client.aclose()

    assert isinstance(image, GeneratedImage)
    assert image.data == expected
    # Тип файла и стоимость берутся из ответа провайдера.
    assert image.media_type == "image/jpeg"
    assert image.cost == 3.756
    assert image.seed == 1001

    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == IMAGE_PATH
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"] == "application/json"

    body = json.loads(request.content.decode())
    assert body == {
        "model": "google/routerai-test-image",
        "prompt": "нарисуй зайца",
        "n": 1,
        "seed": 1001,
        "aspect_ratio": "4:3",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/webp;base64,"
                    + base64.b64encode(reference).decode("ascii")
                },
            }
        ],
    }


async def test_api_key_appears_only_in_authorization_header() -> None:
    """Ключ не подставляется в URL и тело запроса."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler)
    try:
        await provider.generate("промпт", reference_images=[PNG_BYTES])
    finally:
        await http_client.aclose()

    request = captured[0]
    assert API_KEY not in str(request.url)
    assert API_KEY not in request.content.decode()
    assert str(request.url) == IMAGE_PATH


async def test_omits_seed_references_and_aspect_ratio_when_not_given() -> None:
    """Без seed, референсов и соотношения сторон поля не отправляются."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(
        handler, story_image_aspect_ratio=""
    )
    try:
        await provider.generate("только текст")
    finally:
        await http_client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body == {
        "model": "google/routerai-test-image",
        "prompt": "только текст",
        "n": 1,
    }
    # Без seed результат не содержит выдуманного значения.
    assert "seed" not in body
    assert "input_references" not in body
    assert "aspect_ratio" not in body


async def test_sends_one_reference_entry_per_image() -> None:
    """Каждый референс превращается в отдельную ссылку с типом по сигнатуре."""

    captured: list[httpx.Request] = []
    jpeg = JPEG_BYTES + b"photo"
    webp = WEBP_BYTES + b"photo"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler)
    try:
        await provider.generate(
            "промпт", reference_images=[jpeg, webp]
        )
    finally:
        await http_client.aclose()

    references = json.loads(captured[0].content.decode())["input_references"]
    assert references == [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(jpeg).decode("ascii")
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/webp;base64,"
                + base64.b64encode(webp).decode("ascii")
            },
        },
    ]


async def test_unreadable_reference_fails_before_request() -> None:
    """Неопознанные байты референса отклоняются без обращения к сети."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate(
                "промпт", reference_images=[b"GIF89a-musbaka"]
            )
    finally:
        await http_client.aclose()

    assert calls["count"] == 0


async def test_per_call_aspect_ratio_overrides_settings() -> None:
    """Соотношение сторон вызова попадает в тело поверх значения настроек."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler)
    try:
        await provider.generate("лист персонажей", aspect_ratio="1:1")
    finally:
        await http_client.aclose()

    body = json.loads(captured[0].content.decode())
    assert body["aspect_ratio"] == "1:1"


async def test_unsupported_aspect_ratio_of_call_is_rejected() -> None:
    """Соотношение вне списка провайдера отклоняется без обращения к сети."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError) as failure:
            await provider.generate("промпт", aspect_ratio="43:1")
    finally:
        await http_client.aclose()

    assert calls["count"] == 0
    assert "43:1" in str(failure.value)


async def test_media_type_without_prefix_is_normalized() -> None:
    """Тип без префикса вида (`jpeg`) приводится к полному MIME-виду."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_routerai_response(PNG_BYTES, media_type="png")
        )

    provider, http_client = _make_routerai(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.media_type == "image/png"


async def test_missing_media_type_is_detected_by_signature() -> None:
    """Без `media_type` тип определяется по сигнатуре полученных байтов."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_routerai_response(WEBP_BYTES + b"img", media_type=None)
        )

    provider, http_client = _make_routerai(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.media_type == "image/webp"


async def test_missing_cost_is_not_invented() -> None:
    """Без `usage.cost` стоимость остаётся неизвестной."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_routerai_response(PNG_BYTES, cost=None)
        )

    provider, http_client = _make_routerai(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.cost is None


async def test_base_url_with_trailing_slash_is_normalized() -> None:
    """Лишний слеш в базовом URL не даёт двойного слеша в пути."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(
        handler, routerai_base_url=BASE_URL + "/"
    )
    try:
        await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert str(captured[0].url) == IMAGE_PATH


async def test_client_is_lazy_and_aclose_keeps_injected_client() -> None:
    """Клиент создаётся по требованию, а `aclose` не трогает чужой клиент."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    # Без инъекции соединение создаётся только на первом запросе, поэтому
    # закрытие до него — не ошибка.
    lazy_provider = RouterAIImageProvider(_settings())
    assert lazy_provider._client is None
    await lazy_provider.aclose()
    assert lazy_provider._client is None

    provider, http_client = _make_routerai(handler)
    await provider.aclose()

    # Инъектированный клиент остаётся собственностью вызывающего.
    assert provider._client is http_client
    assert not http_client.is_closed
    await http_client.aclose()


# --- Ретраи и пустой ключ ----------------------------------------------


async def test_transient_error_is_retried_then_succeeds() -> None:
    """503 повторяется, и вторая успешная попытка возвращает картинку."""

    calls = {"count": 0}
    expected = JPEG_BYTES + b"after-retry"

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_routerai_response(expected))

    provider, http_client = _make_routerai(handler)
    try:
        image = await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert image.data == expected
    assert calls["count"] == 2


async def test_transient_errors_exhaust_attempts() -> None:
    """Исчерпав попытки, провайдер поднимает транзиентную ошибку."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "boom"})

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(TransientImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_client_errors_are_not_retried(status: int) -> None:
    """Постоянные 4xx падают сразу, ровно один запрос к провайдеру."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(status, json={"error": "nope"})

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 1


@pytest.mark.parametrize("api_key", ["", "   "])
async def test_empty_api_key_fails_before_request(api_key: str) -> None:
    """Пустой или пробельный ключ отменяет вызов без обращения к сети."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_routerai_response(PNG_BYTES))

    provider, http_client = _make_routerai(handler, routerai_api_key=api_key)
    try:
        with pytest.raises(PermanentImageGenerationError) as failure:
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 0
    assert "ROUTERAI_API_KEY" in str(failure.value)


async def test_network_error_is_transient_and_retried() -> None:
    """Сетевая ошибка классифицируется как транзиентная и повторяется."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("нет сети", request=request)

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(TransientImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()

    assert calls["count"] == 3


# --- Учёт стоимости ----------------------------------------------------


async def test_logs_call_cost_and_accumulates_total_without_secret() -> None:
    """Каждый вызов логирует стоимость, сумма считается по числу вызовов."""

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_routerai_response(PNG_BYTES, cost=3.5)
        )

    provider, http_client = _make_routerai(handler)
    try:
        for _ in range(3):
            await provider.generate("промпт")
    finally:
        logger.remove(sink_id)
        await http_client.aclose()

    cost_lines = [line for line in messages if "cost=3.5" in line]
    assert len(cost_lines) == 3
    # Суммарная стоимость растёт по мере вызовов и равна 3 × 3.5.
    assert "total_cost=3.5" in cost_lines[0]
    assert "total_cost=10.5" in cost_lines[-1]
    assert provider.total_cost == 10.5
    # Ключ в лог не попадает ни при каком виде строки.
    assert not any(API_KEY in line for line in messages)


def test_generated_image_carries_scalars_only() -> None:
    """Тип и стоимость сериализуются в JSON и не тащат бинари в артефакт.

    Единственное бинарное поле — сами байты картинки: в `story.json` они не
    попадают, туда уходят пути к файлам. Новые поля — скаляры, поэтому в
    состояние прогона они ложатся без потерь.
    """

    dumped = GeneratedImage(
        data=PNG_BYTES, seed=7, media_type="image/jpeg", cost=3.75
    ).model_dump(mode="json", exclude={"data"})

    assert dumped == {"seed": 7, "media_type": "image/jpeg", "cost": 3.75}


async def test_cost_is_not_invented_when_provider_omits_it() -> None:
    """Без `usage.cost` сумма остаётся неизвестной, а не нулевой."""

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_routerai_response(PNG_BYTES, cost=None)
        )

    provider, http_client = _make_routerai(handler)
    try:
        await provider.generate("промпт")
    finally:
        logger.remove(sink_id)
        await http_client.aclose()

    assert provider.total_cost is None
    # В логе стоимость помечена как неизвестная, а не подменена нулём.
    assert any(
        "cost=unknown" in line and "total_cost=unknown" in line
        for line in messages
    )


# --- Разбор ответа -----------------------------------------------------



@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="no-data"),
        pytest.param({"data": []}, id="empty-data"),
        pytest.param({"data": [{"media_type": "image/png"}]}, id="no-b64"),
        pytest.param({"data": [{"b64_json": ""}]}, id="empty-b64"),
        pytest.param({"data": "не список"}, id="data-not-list"),
    ],
)
async def test_response_without_image_raises_permanent_error(
    payload: dict[str, object],
) -> None:
    """Ответ без картинки поднимает типизированную постоянную ошибку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()


async def test_broken_base64_raises_permanent_error() -> None:
    """Некорректный base64 поднимает типизированную постоянную ошибку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": "не base64 !!!", "media_type": "image/png"}
                ],
                "usage": {"cost": 3.756},
            },
        )

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()


async def test_non_json_response_raises_permanent_error() -> None:
    """Ответ, который не является JSON, поднимает постоянную ошибку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>шлюз ответил прокси-страницей</html>"
        )

    provider, http_client = _make_routerai(handler)
    try:
        with pytest.raises(PermanentImageGenerationError):
            await provider.generate("промпт")
    finally:
        await http_client.aclose()
