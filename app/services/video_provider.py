"""Генерация видео из фото игрушки.

Определяет протокол `VideoProvider` с двумя реализациями:
- `KlingProvider` — реальный асинхронный провайдер (Kling AI через GMI
  Cloud): создание запроса, получение `request_id`, опрос статуса до
  `success`/`failed`, соблюдение максимального ожидания;
- `MockVideoProvider` — заглушка, включаемая при отсутствии API-ключа.

Изображение передаётся в теле запроса как base64-строка, публичный URL не
используется; аутентификация — только в заголовке `Authorization`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

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

PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})

# Нормализация статусов разных диалектов провайдера.
_STATUS_ALIASES: dict[str, str] = {
    "submitted": "queued",
    "queued": "queued",
    "pending": "queued",
    "created": "queued",
    "waiting": "queued",
    "processing": "processing",
    "running": "processing",
    "in_progress": "processing",
    "in-progress": "processing",
    "generating": "processing",
    "success": "success",
    "succeeded": "success",
    "succeed": "success",
    "completed": "success",
    "complete": "success",
    "done": "success",
    "failed": "failed",
    "failure": "failed",
    "error": "failed",
    "rejected": "failed",
}


# --- Ошибки -----------------------------------------------------------------


class VideoGenerationError(Exception):
    """Базовая ошибка генерации видео."""


class TransientVideoError(VideoGenerationError):
    """Временная ошибка — запрос имеет смысл повторить."""


class PermanentVideoError(VideoGenerationError):
    """Нетранзиентная ошибка — повтор не поможет."""


class VideoFailedError(VideoGenerationError):
    """Провайдер сообщил о неуспешном терминальном статусе."""


class VideoTimeoutError(VideoGenerationError):
    """Превышено настроенное максимальное ожидание генерации."""


# --- Модели данных -----------------------------------------------------------


@dataclass(frozen=True)
class VideoJob:
    """Созданный запрос генерации видео."""

    request_id: str


@dataclass(frozen=True)
class VideoResult:
    """Готовое видео."""

    video_url: str


@runtime_checkable
class VideoProvider(Protocol):
    """Интерфейс видеопровайдера."""

    async def create_video(self, image_base64: str, action: str) -> VideoJob:
        """Создать запрос генерации и вернуть его идентификатор."""

    async def wait_for_video(self, request_id: str) -> VideoResult:
        """Опросить статус до успеха/ошибки и вернуть готовое видео."""


# --- Промпт ------------------------------------------------------------------


def build_video_prompt(action: str) -> str:
    """Промпт, производный от выбранного действия игрушки."""

    return (
        f"Оживи игрушку на фотографии: она выполняет действие «{action}». "
        "Плавное движение, кинематографичный стиль, тёплое освещение."
    )


# --- Реальный провайдер ------------------------------------------------------


class KlingProvider:
    """Асинхронный image-to-video провайдер, совместимый с Kling."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        *,
        max_attempts: int = 3,
        retry_wait: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
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
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic

    # --- Публичный API --------------------------------------------------

    async def create_video(self, image_base64: str, action: str) -> VideoJob:
        """Создать запрос генерации видео и вернуть `request_id`."""

        payload = self.build_create_payload(
            image_base64=image_base64,
            action=action,
        )
        request_id = await self._create_with_retries(payload)
        return VideoJob(request_id=request_id)

    def build_create_payload(
        self,
        *,
        image_base64: str,
        action: str,
    ) -> dict[str, Any]:
        """Собрать тело запроса генерации видео.

        Параметры MVP: 5 секунд, стандартный режим, 720p, звук выключен;
        изображение — base64-строкой в теле запроса.
        """

        settings = self._settings
        return {
            "model": settings.kling_model,
            "prompt": build_video_prompt(action),
            "image": image_base64,
            "duration": settings.kling_duration,
            "mode": settings.kling_mode,
            "resolution": settings.kling_resolution,
            "sound": settings.kling_sound,
        }

    async def wait_for_video(self, request_id: str) -> VideoResult:
        """Опросить статус до терминального состояния или таймаута."""

        started = self._clock()
        interval = self._settings.kling_poll_interval
        max_wait = self._settings.kling_max_wait
        observed: str | None = None

        while True:
            status, video_url, raw = await self._fetch_status(request_id)

            if status != observed:
                logger.info(
                    "Kling: статус запроса {} изменился: {} -> {}",
                    request_id,
                    observed or "нет",
                    status,
                )
                observed = status

            if status == "success":
                if not video_url:
                    raise PermanentVideoError(
                        "Провайдер сообщил об успехе, но не вернул URL видео."
                    )
                return VideoResult(video_url=video_url)

            if status == "failed":
                logger.error(
                    "Kling: запрос {} завершился ошибкой: {}",
                    request_id,
                    raw,
                )
                raise VideoFailedError(
                    f"Провайдер не смог сгенерировать видео: {raw}"
                )

            if self._clock() - started >= max_wait:
                raise VideoTimeoutError(
                    "Превышено максимальное ожидание генерации видео "
                    f"({max_wait:g} с)."
                )

            await self._sleep(interval)

    async def aclose(self) -> None:
        """Закрыть собственный HTTP-клиент, если он создавался здесь."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Внутренняя реализация -----------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.kling_timeout
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.kling_api_key}",
            "Content-Type": "application/json",
        }

    async def _create_with_retries(self, payload: dict[str, Any]) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type(TransientVideoError),
            reraise=True,
        ):
            with attempt:
                return await self._post_create(payload)

        raise PermanentVideoError(  # pragma: no cover
            "Не удалось создать запрос генерации видео."
        )

    async def _post_create(self, payload: dict[str, Any]) -> str:
        url = (
            f"{self._settings.kling_base_url.rstrip('/')}"
            f"{self._settings.kling_create_path}"
        )

        try:
            response = await self._build_client().post(
                url, json=payload, headers=self._auth_headers()
            )
        except httpx.TransportError as exc:
            raise TransientVideoError(
                f"Сетевая ошибка при создании видео: {exc}"
            ) from exc

        self._raise_for_status(response, action="создании")
        data = self._parse_json(response, action="создания видео")
        return self._extract_request_id(data)

    async def _fetch_status(
        self,
        request_id: str,
    ) -> tuple[str, str | None, Any]:
        path = self._settings.kling_status_path.format(request_id=request_id)
        url = f"{self._settings.kling_base_url.rstrip('/')}{path}"

        try:
            response = await self._build_client().get(
                url, headers=self._auth_headers()
            )
        except httpx.TransportError as exc:
            raise TransientVideoError(
                f"Сетевая ошибка при опросе статуса видео: {exc}"
            ) from exc

        self._raise_for_status(response, action="опросе статуса")
        data = self._parse_json(response, action="опроса статуса")

        raw_status = self._extract_status(data)
        normalized = _STATUS_ALIASES.get(raw_status.lower(), "")
        if not normalized:
            logger.warning(
                "Kling: неизвестный статус {!r}, продолжаю опрос",
                raw_status,
            )
            normalized = "processing"

        video_url = self._extract_video_url(data)
        return normalized, video_url, data

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        action: str,
    ) -> None:
        if response.status_code in PERMANENT_STATUS_CODES:
            raise PermanentVideoError(
                f"Провайдер отклонил запрос при {action} "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientVideoError(
                f"Провайдер временно недоступен при {action} "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise PermanentVideoError(
                f"Неожиданный ответ провайдера при {action} "
                f"(HTTP {response.status_code})."
            )

    @staticmethod
    def _parse_json(response: httpx.Response, *, action: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentVideoError(
                f"Ответ провайдера при {action} не удалось разобрать."
            ) from exc
        if not isinstance(data, dict):
            raise PermanentVideoError(
                f"Неожиданный формат ответа провайдера при {action}."
            )
        return data

    @staticmethod
    def _extract_request_id(data: dict[str, Any]) -> str:
        candidates: list[Any] = [
            data.get("request_id"),
            data.get("task_id"),
            data.get("id"),
        ]
        nested = data.get("data")
        if isinstance(nested, dict):
            candidates.extend(
                [nested.get("request_id"), nested.get("task_id"), nested.get("id")]
            )

        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
            if isinstance(candidate, int):
                return str(candidate)

        raise PermanentVideoError(
            "Провайдер не вернул идентификатор запроса генерации видео."
        )

    @staticmethod
    def _extract_status(data: dict[str, Any]) -> str:
        candidates: list[Any] = [data.get("status"), data.get("task_status")]
        nested = data.get("data")
        if isinstance(nested, dict):
            candidates.extend([nested.get("status"), nested.get("task_status")])

        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate

        raise PermanentVideoError("Провайдер не вернул статус запроса видео.")

    @staticmethod
    def _extract_video_url(data: dict[str, Any]) -> str | None:
        containers: list[dict[str, Any]] = [data]
        if isinstance(data.get("data"), dict):
            containers.append(data["data"])
        if isinstance(data.get("task_result"), dict):
            containers.append(data["task_result"])

        for container in containers:
            for key in ("video_url", "url", "video"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value

            videos = container.get("videos")
            if isinstance(videos, list) and videos:
                first = videos[0]
                if isinstance(first, dict):
                    for key in ("url", "video_url"):
                        value = first.get(key)
                        if isinstance(value, str) and value:
                            return value

        return None


# --- Заглушка ----------------------------------------------------------------


class MockVideoProvider:
    """Заглушка видеопровайдера: без HTTP-вызовов отдаёт тестовое видео."""

    def __init__(self, settings: Settings) -> None:
        self._video_url = settings.mock_video_url

    async def create_video(self, image_base64: str, action: str) -> VideoJob:
        request_id = f"mock-{uuid4().hex}"
        logger.info(
            "Видеопровайдер не настроен: выдаю заглушку (request_id={})",
            request_id,
        )
        return VideoJob(request_id=request_id)

    async def wait_for_video(self, request_id: str) -> VideoResult:
        logger.info(
            "Заглушка видео: возвращаю тестовое видео для запроса {}",
            request_id,
        )
        return VideoResult(video_url=self._video_url)


# --- Фабрика -----------------------------------------------------------------


def build_video_provider(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> VideoProvider:
    """Выбрать провайдера по конфигурации.

    Без `KLING_API_KEY` возвращается mock-заглушка, сохраняющая тот же
    интерфейс, что и реальный провайдер.
    """

    if not settings.kling_api_key:
        return MockVideoProvider(settings)
    return KlingProvider(settings, client)


__all__ = [
    "KlingProvider",
    "MockVideoProvider",
    "PermanentVideoError",
    "TransientVideoError",
    "VideoFailedError",
    "VideoGenerationError",
    "VideoJob",
    "VideoProvider",
    "VideoResult",
    "VideoTimeoutError",
    "build_video_prompt",
    "build_video_provider",
]
