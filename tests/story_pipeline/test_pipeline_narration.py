"""Тесты узла «иллюстрации и озвучка»: параллельность и прогресс."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.story_pipeline.models import StoryState
from app.story_pipeline.pipeline import (
    IllustrationsAndNarration,
    StoryPipeline,
)
from app.story_pipeline.settings import StorySettings
from tests.story_pipeline.conftest import FakeImageProvider, FakeLLMClient


def state(tmp_path: Path) -> StoryState:
    """Минимальное состояние прогона."""

    return StoryState(
        theme="Тема",
        age=5,
        moral="помогать",
        num_pages=1,
        output_dir=str(tmp_path),
        started_at=datetime.now(timezone.utc),
    )


class HandshakeStage:
    """Этап, который ждёт второго: доказательство параллельного запуска."""

    def __init__(self, name: str, partner: asyncio.Event, me: asyncio.Event) -> None:
        self.name = name
        self._partner = partner
        self._me = me
        self.started = False
        self.finished = False

    async def run(self, state: StoryState) -> StoryState:
        self.started = True
        self._me.set()
        await asyncio.wait_for(self._partner.wait(), timeout=1.0)
        self.finished = True
        return state


@pytest.mark.asyncio
async def test_illustration_and_narration_run_in_parallel(
    tmp_path: Path,
) -> None:
    """Оба этапа стартуют до завершения друг друга и шлют свой прогресс."""

    notified: list[str] = []

    async def notify(name: str) -> None:
        notified.append(name)

    image_ready = asyncio.Event()
    narration_ready = asyncio.Event()
    illustration = HandshakeStage("illustrate", narration_ready, image_ready)
    narration = HandshakeStage("narration", image_ready, narration_ready)

    node = IllustrationsAndNarration(
        illustration, narration, notify=notify
    )
    await node.run(state(tmp_path))

    assert illustration.finished is True
    assert narration.finished is True
    assert notified == ["narration"]


def test_pipeline_without_narration_keeps_single_illustration_stage() -> None:
    """Без провайдера озвучки граф остаётся прежним: один узел иллюстраций."""

    pipeline = StoryPipeline(
        FakeLLMClient([]), FakeImageProvider(), StorySettings(_env_file=None)  # type: ignore[call-arg]
    )
    assert pipeline.stages[-1].name == "illustrate"


class FakeNarrationProvider:
    """Минимальный провайдер озвучки для проверки сборки графа."""

    def __init__(self) -> None:
        self.closed = 0

    async def synthesize(self, text: str) -> bytes:  # pragma: no cover - не вызывается
        raise AssertionError("синтез не должен вызываться в этом тесте")

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_pipeline_closes_narration_provider_when_owned() -> None:
    """Провайдер озвучки закрывается вместе с остальными, если он свой."""

    provider = FakeNarrationProvider()
    pipeline = StoryPipeline(
        FakeLLMClient([]),
        FakeImageProvider(),
        StorySettings(_env_file=None),  # type: ignore[call-arg]
        owns_providers=True,
        narration_provider=provider,
    )

    await pipeline.aclose()

    assert provider.closed == 1
