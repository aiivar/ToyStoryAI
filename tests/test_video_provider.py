"""Тесты сервиса генерации видео (`app/services/video_provider.py`)."""

from __future__ import annotations

import json

import httpx
import pytest
from loguru import logger
from tenacity import wait_none

from app.config import Settings
from app.services.video_provider import (
    KlingProvider,
    MockVideoProvider,
    PermanentVideoError,
    VideoFailedError,
    VideoTimeoutError,
    build_video_prompt,
    build_video_provider,
)


def _settings(**overrides) -> Settings:
    base = {
        "_env_file": None,
        "max_bot_token": "test-token",
        "kling_api_key": "kling-key",
        "kling_poll_interval": 7,
        "kling_max_wait": 600,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _SleepRecorder:
    """Подмена asyncio.sleep: запоминает интервалы, не ждёт."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _status_response(status: str, **extra) -> httpx.Response:
    payload = {"status": status}
    payload.update(extra)
    return httpx.Response(200, json=payload)


# --- Задача 5.2: тело запроса и заголовки ------------------------------------


async def test_create_request_body_and_headers() -> None:
    """Тело запроса содержит ровно параметры MVP и base64-изображение."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"request_id": "req-1"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(_settings(), client, retry_wait=wait_none())
    try:
        await provider.create_video(image_base64="BASE64DATA", action="Танец")
    finally:
        await client.aclose()

    request = captured[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/videos/image2video")
    assert request.headers["Authorization"] == "Bearer kling-key"
    assert request.headers["Content-Type"] == "application/json"
    # Ключ никогда не передаётся в query-параметрах.
    assert request.url.params.get("api_key") is None
    assert request.url.params.get("key") is None

    body = json.loads(request.content.decode())
    assert body == {
        "model": "kling-v3-image-to-video",
        "prompt": build_video_prompt("Танец"),
        "image": "BASE64DATA",
        "duration": "5",
        "mode": "std",
        "resolution": "720p",
        "sound": False,
    }


async def test_prompt_reflects_action() -> None:
    """Промпт описывает действие, плавное движение и тёплый свет."""

    prompt = build_video_prompt("Полёт")
    assert "Полёт" in prompt
    assert "Плавное движение" in prompt
    assert "кинематографичный" in prompt
    assert "тёплое освещение" in prompt


# --- Задача 5.1: жизненный цикл запроса --------------------------------------


async def test_polls_until_success() -> None:
    """Опрос идёт до успеха, интервал берётся из настроек."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": "req-42"})
        if request.method == "GET" and len(requests) == 2:
            return _status_response("processing")
        return _status_response(
            "success", video_url="https://cdn.example.com/v.mp4"
        )

    sleep = _SleepRecorder()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(
        _settings(),
        client,
        retry_wait=wait_none(),
        sleep=sleep,
    )
    try:
        job = await provider.create_video("BASE64", "Путешествие")
        result = await provider.wait_for_video(job.request_id)
    finally:
        await client.aclose()

    assert job.request_id == "req-42"
    assert result.video_url == "https://cdn.example.com/v.mp4"
    assert sleep.calls == [7.0]

    status_requests = [r for r in requests if r.method == "GET"]
    assert status_requests[0].url.path.endswith("/videos/image2video/req-42")


async def test_failed_status_raises_error() -> None:
    """Неуспешный статус поднимает ошибку генерации видео."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": "req-x"})
        return _status_response("failed", error="provider exploded")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(
        _settings(), client, retry_wait=wait_none(), sleep=_SleepRecorder()
    )
    try:
        job = await provider.create_video("BASE64", "Танец")
        with pytest.raises(VideoFailedError):
            await provider.wait_for_video(job.request_id)
    finally:
        await client.aclose()


async def test_timeout_when_status_stays_queued() -> None:
    """Превышение максимального ожидания завершает попытку таймаутом."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": "req-slow"})
        return _status_response("queued")

    ticks = iter([0.0, 999.0, 999.0])

    def clock() -> float:
        return next(ticks, 999.0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(
        _settings(),
        client,
        retry_wait=wait_none(),
        sleep=_SleepRecorder(),
        clock=clock,
    )
    try:
        job = await provider.create_video("BASE64", "Танец")
        with pytest.raises(VideoTimeoutError):
            await provider.wait_for_video(job.request_id)
    finally:
        await client.aclose()


async def test_transient_create_error_is_retried() -> None:
    """Временный сбой создания повторяется и затем завершается успехом."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json={"request_id": "req-ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(_settings(), client, retry_wait=wait_none())
    try:
        job = await provider.create_video("BASE64", "Танец")
    finally:
        await client.aclose()

    assert calls["count"] == 2
    assert job.request_id == "req-ok"


async def test_auth_error_is_not_retried() -> None:
    """401 при создании не повторяется."""

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = KlingProvider(_settings(), client, retry_wait=wait_none())
    try:
        with pytest.raises(PermanentVideoError):
            await provider.create_video("BASE64", "Танец")
    finally:
        await client.aclose()

    assert calls["count"] == 1


# --- Задача 5.4: логирование переходов --------------------------------------


async def test_logs_status_transitions() -> None:
    """Каждый наблюдаемый переход статуса попадает в лог."""

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": "req-log"})
        options = {
            2: _status_response("queued"),
            3: _status_response("processing"),
        }
        return options.get(
            len(requests), _status_response("success", video_url="https://v/1.mp4")
        )

    requests: list[httpx.Request] = []
    original = handler

    def counting_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return original(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(counting_handler))
    provider = KlingProvider(
        _settings(), client, retry_wait=wait_none(), sleep=_SleepRecorder()
    )
    try:
        job = await provider.create_video("BASE64", "Полёт")
        await provider.wait_for_video(job.request_id)
    finally:
        await client.aclose()
        logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "queued" in joined
    assert "processing" in joined
    assert "success" in joined


# --- Задача 5.3: mock-заглушка -----------------------------------------------


async def test_mock_provider_returns_test_video_without_http(
    settings: Settings,
) -> None:
    """Без API-ключа возвращается ссылка на тестовое видео без HTTP-вызова."""

    provider = build_video_provider(settings)

    assert isinstance(provider, MockVideoProvider)
    assert not hasattr(provider, "_client")

    job = await provider.create_video("BASE64", "Танец")
    result = await provider.wait_for_video(job.request_id)

    assert job.request_id.startswith("mock-")
    assert result.video_url == settings.mock_video_url


async def test_factory_selects_kling_with_api_key() -> None:
    """При заданном ключе выбирается реальный провайдер."""

    provider = build_video_provider(_settings())

    assert isinstance(provider, KlingProvider)
