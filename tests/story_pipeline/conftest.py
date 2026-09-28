"""Общие фикстуры тестов пайплайна сказок.

Этапы принимают зависимости (`LLMClient`, `ImageProvider`) через
конструктор, поэтому в тестах подставляются `FakeLLMClient` и
`FakeImageProvider` — реализации протоколов без сети. Фейки записывают все
вызовы и по очереди отдают заранее подготовленные ответы.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from app.story_pipeline.providers.image import GeneratedImage
from app.story_pipeline.providers.llm import Usage
from app.story_pipeline.settings import StorySettings


class FakeLLMClient:
    """Фейковый `LLMClient`: отдаёт ответы по очереди и пишет вызовы."""

    def __init__(
        self,
        responses: Sequence[Any],
        *,
        usage: Usage | None = None,
    ) -> None:
        self._responses = list(responses)
        self._usage = usage or Usage(prompt_tokens=11, completion_tokens=22)
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Сколько раз этап обратился к модели."""

        return len(self.calls)

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float,
    ) -> tuple[Any, Usage]:
        """Записать вызов и вернуть следующий заготовленный ответ."""

        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response, self._usage


class FakeImageProvider:
    """Фейковый `ImageProvider`: отдаёт ответы по очереди и пишет вызовы.

    Без `responses` детерминированно строит ответ из номера вызова, чтобы
    сквозной прогон не зависел от заготовок. Метода `aclose` умышленно нет:
    так проверяется безопасное закрытие провайдера без закрытия.
    """

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self._responses = list(responses) if responses is not None else None
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Записать вызов и вернуть следующий заготовленный ответ."""

        index = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "seed": seed,
                "reference_images": reference_images,
                "aspect_ratio": aspect_ratio,
            }
        )
        if self._responses is not None:
            response = self._responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return GeneratedImage(
            data=f"image-{index}".encode("ascii"), seed=100 + index
        )


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLMClient]:
    """Фабрика фейкового `LLMClient` для конкретного сценария теста."""

    def factory(
        responses: Sequence[Any], *, usage: Usage | None = None
    ) -> FakeLLMClient:
        return FakeLLMClient(responses, usage=usage)

    return factory


@pytest.fixture
def fake_image_provider() -> Callable[..., FakeImageProvider]:
    """Фабрика фейкового `ImageProvider` для конкретного сценария теста."""

    def factory(responses: Sequence[Any] | None = None) -> FakeImageProvider:
        return FakeImageProvider(responses)

    return factory


@pytest.fixture
def story_settings() -> Callable[..., StorySettings]:
    """Фабрика `StorySettings`, изолированная от локального `.env` и окружения.

    Значения передаются явно как аргументы конструктора: они имеют приоритет
    над переменными окружения, поэтому тесты герметичны.
    """

    def factory(**overrides: Any) -> StorySettings:
        # `_env_file=None` отключает чтение локального `.env`.
        return StorySettings(  # type: ignore[call-arg]
            _env_file=None, **overrides
        )

    return factory
