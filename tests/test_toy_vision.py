"""Тесты сервиса распознавания игрушки (`app/services/toy_vision.py`).

Сервис — источник правды об образе героя: описание игрушки должно совпадать
с фото, поэтому проверяются и форма запроса (байты фото в data URI, низкая
температура, JSON-режим), и отказы (одна повторная попытка, постоянные
ошибки без повторов, пустой ключ до сети), и защита ключа.

Тесты герметичны: внешние вызовы подменены `httpx.MockTransport`, живой сети
нет. Попытка повтора отключена через `wait_none()`, поэтому тесты быстрые.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from app.config import Settings
from app.services.toy_vision import (
    TOY_VISION_PROMPT,
    PermanentToyVisionError,
    ToyDescription,
    ToyVisionError,
    ToyVisionService,
    TransientToyVisionError,
)

# Минимальные заголовки реальных файлов: подпись плюс хвост, который
# отличает WebP (RIFF-контейнер) от прочих RIFF-файлов.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00"
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00"

TEST_API_KEY = "SECRET-DEEPSEEK-KEY"

TOY_JSON = {
    "kind": "плюшевый медвежонок",
    "color": "рыжий с белой грудкой",
    "material": "мягкая ткань и синтепон",
    "features": "круглые уши, вышитый шарф, короткие лапы",
    "short": "рыжий плюшевый мишка в шарфе",
}

TOY_RESPONSE = {
    "choices": [
        {"message": {"content": json.dumps(TOY_JSON, ensure_ascii=False)}}
    ]
}


def _make_settings(**overrides: object) -> Settings:
    """Настройки бота с тестовым ключом DeepSeek."""

    values: dict[str, object] = {
        "max_bot_token": "test-token",
        "deepseek_api_key": TEST_API_KEY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _make_service(
    handler: Any,
    **overrides: object,
) -> tuple[ToyVisionService, httpx.AsyncClient]:
    """Сервис на `MockTransport` без ожидания между попытками."""

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = ToyVisionService(
        _make_settings(**overrides), client, retry_wait=wait_none()
    )
    return service, client


def _body(request: httpx.Request) -> dict[str, Any]:
    """Разобрать тело исходящего запроса."""

    payload: dict[str, Any] = json.loads(request.content.decode())
    return payload


# --- Задача 2.2: вызов DeepSeek с изображением ------------------------------


async def test_sends_photo_as_data_uri_and_parses_description() -> None:
    """Один вызов DeepSeek получает фото в data URI и отдаёт описание."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        description = await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert description == ToyDescription(**TOY_JSON)

    assert len(captured) == 1
    request = captured[0]
    body = _body(request)
    parts = body["messages"][0]["content"]

    assert request.url.path.endswith("/chat/completions")
    assert request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
    assert body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == _make_settings().deepseek_max_tokens
    assert body["messages"][0]["role"] == "user"
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == TOY_VISION_PROMPT
    assert parts[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/webp;base64,"
            f"{base64.b64encode(WEBP_BYTES).decode('ascii')}"
        },
    }


async def test_markdown_fenced_json_is_parsed() -> None:
    """JSON в markdown-обёртке разбирается, как у LLM-клиента пайплайна."""

    fenced = f"```json\n{json.dumps(TOY_JSON, ensure_ascii=False)}\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": fenced}}]}
        )

    service, client = _make_service(handler)
    try:
        description = await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert description.short == "рыжий плюшевый мишка в шарфе"


async def test_prompt_demands_image_only_and_brief_output() -> None:
    """Задание модели требует опираться только на фото и отвечать кратко."""

    prompt = TOY_VISION_PROMPT.lower()

    assert "только на то, что видно" in prompt
    assert "не додумывай" in prompt
    assert "кратко" in prompt
    for field in ("kind", "color", "material", "features", "short"):
        assert field in TOY_VISION_PROMPT


# --- Задача 2.4: MIME по сигнатуре ------------------------------------------


@pytest.mark.parametrize(
    ("data", "mime_type"),
    [
        pytest.param(JPEG_BYTES, "image/jpeg", id="jpeg"),
        pytest.param(PNG_BYTES, "image/png", id="png"),
        pytest.param(WEBP_BYTES, "image/webp", id="webp"),
    ],
)
async def test_data_uri_mime_type_matches_signature(
    data: bytes,
    mime_type: str,
) -> None:
    """Тип в data URI определяется по содержимому байтов, а не по имени."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        await service.describe(data)
    finally:
        await client.aclose()

    parts = _body(captured[0])["messages"][0]["content"]
    url = parts[1]["image_url"]["url"]
    assert url == (
        f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
    )


async def test_unknown_photo_bytes_are_rejected_before_request() -> None:
    """Неопознанные байты не уходят в сеть молчаливым мусором."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(b"GIF89a\x01\x00\x01\x00")
    finally:
        await client.aclose()

    assert calls == []
    message = str(exc_info.value)
    assert "JPEG" in message and "WebP" in message


# --- Задача 2.3: одна повторная попытка -------------------------------------


async def test_transient_failure_is_retried_once() -> None:
    """Первый временный сбой (5xx) переживает один повтор и не срывает флоу."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        description = await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert len(calls) == 2
    assert description.kind == "плюшевый медвежонок"


async def test_transient_failure_is_not_retried_more_than_once() -> None:
    """Второй временный сбой поднимает ошибку, попыток ровно две."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, text="rate limit")

    service, client = _make_service(handler)
    try:
        with pytest.raises(TransientToyVisionError):
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert len(calls) == 2


async def test_transport_error_is_retried_once() -> None:
    """Сетевой обрыв (httpx.TransportError) считается временной ошибкой."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        description = await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert len(calls) == 2
    assert description.kind == "плюшевый медвежонок"


# --- Задача 2.3: постоянные отказы без повторов ----------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_status_raises_without_retry(status: int) -> None:
    """Постоянный HTTP-статус сразу даёт PermanentToyVisionError (1 попытка).

    400 и 422 покрывают в том числе отказ модели принимать изображения.
    """

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "nope"}})

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert str(status) in str(exc_info.value)


async def test_model_without_image_support_fails_with_clear_error() -> None:
    """Отказ «vision не поддерживается» даёт понятный текст, а не сказку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": (
                        "This model does not support image input, "
                        "please use a vision model."
                    )
                }
            },
        )

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    message = str(exc_info.value)
    assert "HTTP 400" in message
    # Причина отказа вендора видна в тексте, а стектрейса нет.
    assert "does not support image input" in message
    assert "Traceback" not in message


# --- Задача 2.3: ответ не по схеме -----------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("не JSON вовсе", id="not-json"),
        pytest.param('{"kind": "мишка", "color": "рыжий"}', id="missing"),
        pytest.param(
            '{"kind": 1, "color": "рыжий", "material": "ткань", '
            '"features": "уши", "short": "мишка"}',
            id="wrong-types",
        ),
    ],
)
async def test_response_off_schema_raises_typed_error(
    content: str,
) -> None:
    """Ответ не по схеме — типизированная ошибка, без выдуманного описания."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError):
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    # Схема описания не размывается дефолтами: все поля обязательны, иначе
    # модель могла бы не прислать, например, материал, и мы бы выдумали его.
    assert len(calls) == 1
    for field in ("kind", "color", "material", "features", "short"):
        assert ToyDescription.model_fields[field].is_required()


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        pytest.param({}, "отсутствует", id="empty-envelope"),
        pytest.param({"choices": []}, "отсутствует", id="no-choices"),
        pytest.param(
            {"choices": [{"message": {"content": "   "}}]},
            "пустое",
            id="empty-content",
        ),
        pytest.param(
            {"choices": [{"message": {}}]}, "отсутствует", id="no-content"
        ),
        pytest.param(
            {"choices": [{"message": {"content": 42}}]},
            "пустое",
            id="non-string-content",
        ),
    ],
)
async def test_broken_envelope_raises_typed_error(
    payload: dict[str, Any],
    fragment: str,
) -> None:
    """Ответ без нормального конверта Chat Completions — тоже ошибка."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert fragment in str(exc_info.value)


async def test_unparsable_json_response_raises_typed_error() -> None:
    """Ответ, который не разбирается как JSON, даёт понятную ошибку."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    service, client = _make_service(handler)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert "разобрать" in str(exc_info.value)


# --- Задача 2.3: пустой ключ и защита секрета -------------------------------


@pytest.mark.parametrize("key", ["", "   "])
async def test_empty_api_key_fails_before_network(key: str) -> None:
    """Пустой ключ — понятная ошибка без единого сетевого вызова."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler, deepseek_api_key=key)
    try:
        with pytest.raises(PermanentToyVisionError) as exc_info:
            await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert calls == []
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize(
    ("status", "raises"),
    [
        pytest.param(200, False, id="success"),
        pytest.param(401, True, id="permanent-error"),
        pytest.param(503, True, id="transient-error"),
    ],
)
async def test_api_key_never_leaks_into_body_url_or_errors(
    status: int,
    raises: bool,
) -> None:
    """Ключ уходит только в заголовок: ни в тело, ни в URL, ни в лог."""

    from loguru import logger

    captured: list[httpx.Request] = []
    log_lines: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if status == 200:
            return httpx.Response(200, json=TOY_RESPONSE)
        return httpx.Response(
            status, json={"error": {"message": "service failure"}}
        )

    sink_id = logger.add(
        lambda message: log_lines.append(str(message)), level="DEBUG"
    )
    service, client = _make_service(handler)
    try:
        if raises:
            with pytest.raises(ToyVisionError) as exc_info:
                await service.describe(WEBP_BYTES)
            error_text = str(exc_info.value)
        else:
            await service.describe(WEBP_BYTES)
            error_text = ""
    finally:
        await client.aclose()
        logger.remove(sink_id)

    request = captured[0]
    assert TEST_API_KEY in request.headers["Authorization"]
    assert TEST_API_KEY not in request.content.decode()
    assert TEST_API_KEY not in str(request.url)
    assert TEST_API_KEY not in error_text
    assert all(TEST_API_KEY not in line for line in log_lines)


# --- Задача 2.5: та же модель, что и для текста -----------------------------


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
async def test_vision_uses_same_model_setting_as_text(model: str) -> None:
    """Распознавание идёт той же моделью, что и текст: новой настройки нет."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler, deepseek_model=model)
    try:
        await service.describe(WEBP_BYTES)
    finally:
        await client.aclose()

    assert _body(captured[0])["model"] == model
    assert not any(
        "vision" in name or "toy" in name for name in Settings.model_fields
    )


async def test_aclose_is_safe_for_injected_client() -> None:
    """`aclose()` не закрывает чужой клиент (его закрывает владелец)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TOY_RESPONSE)

    service, client = _make_service(handler)
    try:
        await service.describe(WEBP_BYTES)
        await service.aclose()
        # Инъектированный клиент остаётся рабочим после `aclose()`.
        again = await service.describe(WEBP_BYTES)
        assert again.kind == "плюшевый медвежонок"
    finally:
        await client.aclose()
