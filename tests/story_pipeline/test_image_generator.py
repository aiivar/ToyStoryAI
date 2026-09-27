"""Тесты этапа генерации изображений (`stages/image_generator.py`).

Этап принимает `ImageProvider` через конструктор, поэтому в тестах
подставляется `FakeImageProvider` — реализация протокола без сети. Она
записывает все вызовы и по очереди отдаёт заготовленные ответы или
поднимает заготовленные ошибки.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    TransientImageGenerationError,
)
from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    KeyObject,
    Location,
    PageScene,
    StoryState,
)
from app.story_pipeline.providers.image import (
    GeneratedImage,
    MockImageProvider,
)
from app.story_pipeline.stages.image_generator import ImageGeneratorStage
from app.story_pipeline.stages.prompt_builder import (
    build_art_prompt,
    build_character_sheet_prompt,
)


class FakeImageProvider:
    """Фейковый `ImageProvider`: отдаёт ответы по очереди и пишет вызовы."""

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self._responses = list(responses) if responses is not None else None
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        """Записать вызов и вернуть следующий заготовленный ответ."""

        index = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "seed": seed,
                "reference_images": reference_images,
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


def _entities() -> ExtractedEntities:
    """Реестр сущностей с одним персонажем, локацией и объектом."""

    return ExtractedEntities(
        characters=[
            Character(
                name="Зайка",
                appearance="маленький серый заяц",
                clothing="синий комбинезон",
                distinguishing_features="одно ухо в пятнах",
            )
        ],
        locations=[Location(name="Лес", description="густой еловый лес")],
        key_objects=[KeyObject(name="Мяч", description="красный мяч")],
        art_style="мягкая акварель",
    )


def _scene(page_num: int) -> PageScene:
    """Сцена страницы с одним персонажем, локацией и объектом."""

    return PageScene(
        page_num=page_num,
        scene_description=f"сцена номер {page_num}",
        characters_in_scene=["Зайка"],
        location="Лес",
        objects_in_scene=["Мяч"],
    )


def _state(output_dir: Path, num_scenes: int = 2) -> StoryState:
    """Состояние с реестром, сценами и заданным каталогом вывода."""

    return StoryState(
        theme="дружба",
        age=5,
        moral="помогай друзьям",
        entities=_entities(),
        scenes=[_scene(n) for n in range(1, num_scenes + 1)],
        output_dir=str(output_dir),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


async def test_generates_sheet_first_then_pages_with_reference(
    tmp_path: Path,
) -> None:
    """Сначала character sheet, затем страницы с ним как reference."""

    state = _state(tmp_path)
    entities = state.entities
    assert entities is not None
    provider = FakeImageProvider()

    await ImageGeneratorStage(provider).run(state)

    # Первый вызов — character sheet, без reference.
    sheet_call = provider.calls[0]
    assert sheet_call["prompt"] == build_character_sheet_prompt(entities)
    assert sheet_call["reference_images"] is None

    # Страницы идут в порядке сцен, с промптами из build_art_prompt.
    expected_prompts = [
        build_art_prompt(scene, entities) for scene in state.scenes
    ]
    page_calls = provider.calls[1:]
    assert [call["prompt"] for call in page_calls] == expected_prompts
    assert state.art_prompts == expected_prompts

    # Файлы созданы, sheet передан как reference каждой странице.
    sheet_path = tmp_path / "character_sheet.png"
    sheet_bytes = sheet_path.read_bytes()
    assert sheet_bytes == b"image-0"
    assert (tmp_path / "page_1.png").read_bytes() == b"image-1"
    assert (tmp_path / "page_2.png").read_bytes() == b"image-2"
    for call in page_calls:
        assert call["reference_images"] == [sheet_bytes]

    # Пути и seed записаны в состояние.
    assert state.character_sheet_path == str(sheet_path)
    assert state.character_sheet_seed == 100
    assert [art.page_num for art in state.images] == [1, 2]
    assert [art.path for art in state.images] == [
        str(tmp_path / "page_1.png"),
        str(tmp_path / "page_2.png"),
    ]
    assert [art.seed for art in state.images] == [101, 102]

    # Метрика этапа добавлена, статус успешного прогона не тронут.
    metric = state.metrics[-1]
    assert metric.stage == "illustrate"
    assert metric.attempts == 1
    assert metric.error is None
    assert state.status == "ok"


async def test_page_failure_degrades_to_needs_review(tmp_path: Path) -> None:
    """Сбойная страница не роняет прогон: остальные создаются."""

    state = _state(tmp_path)
    provider = FakeImageProvider(
        responses=[
            GeneratedImage(data=b"sheet", seed=5),
            TransientImageGenerationError("таймаут"),
            GeneratedImage(data=b"page-2", seed=7),
        ]
    )

    await ImageGeneratorStage(provider).run(state)

    # Первая страница отсутствует, вторая создана; прогон не упал.
    assert not (tmp_path / "page_1.png").exists()
    assert (tmp_path / "page_2.png").read_bytes() == b"page-2"
    assert [(art.page_num, art.seed) for art in state.images] == [(2, 7)]
    assert state.character_sheet_path is not None
    assert state.status == "needs_review"

    # Отсутствие зафиксировано в метрике и не затирает seed листа.
    metric = state.metrics[-1]
    assert metric.stage == "illustrate"
    assert metric.error is not None
    assert "1" in metric.error


async def test_permanent_error_on_page_aborts_without_continuing(
    tmp_path: Path,
) -> None:
    """Постоянный сбой страницы останавливает прогон понятной ошибкой."""

    state = _state(tmp_path)
    provider = FakeImageProvider(
        responses=[
            GeneratedImage(data=b"sheet", seed=5),
            PermanentImageGenerationError("невалидный ключ"),
        ]
    )

    with pytest.raises(PermanentImageGenerationError):
        await ImageGeneratorStage(provider).run(state)

    # Вторая страница вслепую не запрашивается, статус не подменяется.
    assert len(provider.calls) == 2
    assert state.status == "ok"
    assert not (tmp_path / "page_2.png").exists()


async def test_sheet_failure_aborts_before_pages(tmp_path: Path) -> None:
    """Сбой character sheet не позволяет рисовать страницы без reference."""

    state = _state(tmp_path)
    provider = FakeImageProvider(
        responses=[TransientImageGenerationError("таймаут")]
    )

    with pytest.raises(TransientImageGenerationError):
        await ImageGeneratorStage(provider).run(state)

    assert len(provider.calls) == 1
    assert not (tmp_path / "page_1.png").exists()
    assert state.status == "ok"


async def test_missing_output_dir_raises_configuration_error() -> None:
    """Без каталога вывода этап падает понятной ошибкой конфигурации."""

    state = _state(Path("."))
    state.output_dir = None

    with pytest.raises(PermanentImageGenerationError):
        await ImageGeneratorStage(FakeImageProvider()).run(state)


async def test_stage_with_real_mock_writes_valid_pngs(tmp_path: Path) -> None:
    """С реальным mock-провайдером на диск пишутся валидные PNG."""

    signature = b"\x89PNG\r\n\x1a\n"
    state = _state(tmp_path)

    await ImageGeneratorStage(MockImageProvider()).run(state)

    sheet_bytes = (tmp_path / "character_sheet.png").read_bytes()
    assert sheet_bytes.startswith(signature)
    assert state.character_sheet_path == str(tmp_path / "character_sheet.png")
    assert len(state.images) == 2
    for art in state.images:
        assert Path(art.path).read_bytes().startswith(signature)
    assert state.status == "ok"
