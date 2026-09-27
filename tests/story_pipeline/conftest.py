"""Общие фикстуры тестов пайплайна сказок.

Этапы принимают зависимость `LLMClient` через конструктор, поэтому в тестах
подставляется `FakeLLMClient` — реализация протокола без сети. Фейк
записывает все вызовы и по очереди отдаёт заранее подготовленные ответы.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from app.story_pipeline.providers.llm import Usage


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


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLMClient]:
    """Фабрика фейкового `LLMClient` для конкретного сценария теста."""

    def factory(
        responses: Sequence[Any], *, usage: Usage | None = None
    ) -> FakeLLMClient:
        return FakeLLMClient(responses, usage=usage)

    return factory
