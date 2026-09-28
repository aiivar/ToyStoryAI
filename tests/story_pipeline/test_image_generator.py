"""Тесты этапа генерации изображений (`stages/image_generator.py`).

Этап принимает `ImageProvider` через конструктор, поэтому в тестах
подставляется общий `FakeImageProvider` из `conftest.py` (фикстура
`fake_image_provider`) — реализация протокола без сети.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
from app.story_pipeline.stages.image_generator import (
    CHARACTER_SHEET_ASPECT_RATIO,
    CHARACTER_SHEET_SEED,
    ImageGeneratorStage,
    page_filename,
    page_seed,
)
from app.story_pipeline.stages.prompt_builder import (
    build_art_prompt,
    build_character_sheet_prompt,
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
        num_pages=num_scenes,
        entities=_entities(),
        scenes=[_scene(n) for n in range(1, num_scenes + 1)],
        output_dir=str(output_dir),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


async def test_generates_sheet_first_then_pages_with_reference(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Сначала character sheet, затем страницы с ним как reference."""

    state = _state(tmp_path)
    entities = state.entities
    assert entities is not None
    provider = fake_image_provider()

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


async def test_photo_reference_builds_sheet_and_sheet_builds_pages(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Фото — единственный референс листа, лист — единственный референс страниц."""

    photo = b"photo-bytes"
    provider = fake_image_provider()

    await ImageGeneratorStage(
        provider, reference_photo=photo
    ).run(_state(tmp_path))

    sheet_call = provider.calls[0]
    page_calls = provider.calls[1:]
    # Ровно один референс на вызов: фото в лист, лист — в страницы.
    assert sheet_call["reference_images"] == [photo]
    for call in page_calls:
        assert call["reference_images"] == [b"image-0"]


async def test_aspect_ratios_differ_between_sheet_and_pages(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Лист персонажей рисуется квадратом, страницы — в пропорции прогона."""

    provider = fake_image_provider()

    await ImageGeneratorStage(
        provider, aspect_ratio="4:3"
    ).run(_state(tmp_path))

    assert provider.calls[0]["aspect_ratio"] == CHARACTER_SHEET_ASPECT_RATIO
    assert [call["aspect_ratio"] for call in provider.calls[1:]] == ["4:3", "4:3"]


async def test_files_are_saved_with_extension_of_returned_media_type(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Имя файла следует типу, который вернул провайдер, без конвертации.

    Живая проверка RouterAI отдаёт JPEG, поэтому жёсткое `.png` приводило к
    файлу с расширением, не совпадающим с содержимым.
    """

    jpeg = b"\xff\xd8\xff-jpeg-bytes"
    png = b"\x89PNG\r\n\x1a\n-png-bytes"
    provider = fake_image_provider(
        responses=[
            GeneratedImage(data=jpeg, seed=1, media_type="image/jpeg"),
            GeneratedImage(data=jpeg, seed=2, media_type="image/jpeg"),
            GeneratedImage(data=png, seed=3, media_type="image/png"),
        ]
    )
    state = _state(tmp_path)

    await ImageGeneratorStage(provider).run(state)

    # Расширение листа и страниц соответствует типу файла.
    assert (tmp_path / "character_sheet.jpg").read_bytes() == jpeg
    assert (tmp_path / "page_1.jpg").read_bytes() == jpeg
    assert (tmp_path / "page_2.png").read_bytes() == png
    # Байты не перекодировались, пути в состоянии совпадают с файлами.
    assert state.character_sheet_path == str(tmp_path / "character_sheet.jpg")
    assert [art.path for art in state.images] == [
        str(tmp_path / "page_1.jpg"),
        str(tmp_path / "page_2.png"),
    ]
    assert state.status == "ok"


async def test_webp_media_type_maps_to_webp_extension(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """WebP от провайдера сохраняется как `.webp`, а не переименовывается."""

    provider = fake_image_provider(
        responses=[
            GeneratedImage(data=b"RIFF", seed=1, media_type="image/webp"),
            GeneratedImage(data=b"RIFF", seed=2, media_type="image/webp"),
        ]
    )

    await ImageGeneratorStage(provider).run(_state(tmp_path, num_scenes=1))

    assert (tmp_path / "character_sheet.webp").read_bytes() == b"RIFF"
    assert (tmp_path / "page_1.webp").read_bytes() == b"RIFF"


async def test_missing_media_type_falls_back_to_png_name(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Пустой тип не роняет прогон: имя строится по разумному умолчанию."""

    provider = fake_image_provider(
        responses=[
            GeneratedImage(data=b"sheet", seed=1, media_type=""),
            GeneratedImage(data=b"page", seed=2, media_type="  "),
        ]
    )
    state = _state(tmp_path, num_scenes=1)

    await ImageGeneratorStage(provider).run(state)

    assert (tmp_path / "character_sheet.png").read_bytes() == b"sheet"
    assert (tmp_path / "page_1.png").read_bytes() == b"page"
    assert state.status == "ok"
    assert state.metrics[-1].error is None


def test_extension_helpers_map_media_types_without_path_separators() -> None:
    """Хелперы имени файла устойчивы к пустому и неожиданному типу."""

    assert page_filename(3, "image/jpeg") == "page_3.jpg"
    assert page_filename(3, "image/webp") == "page_3.webp"
    assert page_filename(3, "image/png") == "page_3.png"
    # Тип неизвестен или пуст — безопасное имя без слешей и точек-обманок.
    assert page_filename(3, None) == "page_3.png"
    assert page_filename(3, "") == "page_3.png"
    # Тип вне таблицы: в имя попадает только подтип, служебные символы нет.
    hostile = page_filename(3, "../../evil")
    assert "/" not in hostile
    assert ".." not in hostile
    assert hostile == "page_3.evil"


async def test_page_failure_degrades_to_needs_review(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Сбойная страница не роняет прогон: остальные создаются."""

    state = _state(tmp_path)
    provider = fake_image_provider(
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
    fake_image_provider,
) -> None:
    """Постоянный сбой страницы останавливает прогон понятной ошибкой."""

    state = _state(tmp_path)
    provider = fake_image_provider(
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


async def test_sheet_failure_aborts_before_pages(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Сбой character sheet не позволяет рисовать страницы без reference."""

    state = _state(tmp_path)
    provider = fake_image_provider(
        responses=[TransientImageGenerationError("таймаут")]
    )

    with pytest.raises(TransientImageGenerationError):
        await ImageGeneratorStage(provider).run(state)

    assert len(provider.calls) == 1
    assert not (tmp_path / "page_1.png").exists()
    assert state.status == "ok"


async def test_missing_output_dir_raises_configuration_error(
    fake_image_provider,
) -> None:
    """Без каталога вывода этап падает понятной ошибкой конфигурации."""

    state = _state(Path("."))
    state.output_dir = None

    with pytest.raises(PermanentImageGenerationError):
        await ImageGeneratorStage(fake_image_provider()).run(state)


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


# --- Seed генерации ------------------------------------------------------


def test_page_seed_is_deterministic_and_distinct_per_page() -> None:
    """Seed страницы выводится из её номера: стабилен и различен у соседей."""

    assert page_seed(1) == page_seed(1)
    assert page_seed(1) != page_seed(2)
    # Номера страниц не влияют на seed соседних прогонов друг друга.
    assert page_seed(2) == page_seed(2)


async def test_provider_receives_distinct_seed_per_page(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Каждая страница и лист персонажей запрашиваются со своим seed."""

    state = _state(tmp_path, num_scenes=3)
    provider = fake_image_provider()

    await ImageGeneratorStage(provider).run(state)

    seeds = [call["seed"] for call in provider.calls]
    assert seeds[0] == CHARACTER_SHEET_SEED
    assert seeds[1:] == [page_seed(1), page_seed(2), page_seed(3)]
    assert len(set(seeds[1:])) == 3


async def test_page_art_records_requested_seed(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Seed провайдера пишется в `PageArt` и в seed character sheet."""

    state = _state(tmp_path, num_scenes=2)
    provider = fake_image_provider(
        responses=[
            GeneratedImage(data=b"sheet", seed=CHARACTER_SHEET_SEED),
            GeneratedImage(data=b"page-1", seed=page_seed(1)),
            GeneratedImage(data=b"page-2", seed=page_seed(2)),
        ]
    )

    await ImageGeneratorStage(provider).run(state)

    assert state.character_sheet_seed == CHARACTER_SHEET_SEED
    assert [art.seed for art in state.images] == [
        page_seed(1),
        page_seed(2),
    ]
    # Ни один seed не остался пустым: поле заполнено для всех страниц.
    assert all(art.seed is not None for art in state.images)


async def test_mock_pages_are_byte_distinct(tmp_path: Path) -> None:
    """В mock-режиме страницы не совпадают байт-в-байт."""

    state = _state(tmp_path, num_scenes=3)

    await ImageGeneratorStage(MockImageProvider()).run(state)

    payloads = [Path(art.path).read_bytes() for art in state.images]
    assert len(set(payloads)) == len(payloads)
    # Лист персонажей не совпадает ни с одной страницей.
    sheet = (tmp_path / "character_sheet.png").read_bytes()
    assert sheet not in payloads


# --- Контроль полноты покрытия страниц ----------------------------------


async def test_missing_page_png_marks_needs_review(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Прогон не может завершиться `ok`, если файл какой-то страницы пропал.

    Регрессия на исходный дефект: сцены для страниц 1 и 3 при трёх
    страницах давали `character_sheet.png`, `page_1.png`, `page_3.png` и
    итоговый `status == "ok"` — файл `page_2.png` просто отсутствовал.
    """

    state = _state(tmp_path, num_scenes=3)
    state.scenes = [_scene(1), _scene(3)]

    await ImageGeneratorStage(fake_image_provider()).run(state)

    assert not (tmp_path / "page_2.png").exists()
    assert state.status == "needs_review"
    metric = state.metrics[-1]
    assert metric.stage == "illustrate"
    assert metric.error is not None
    assert "2" in metric.error


async def test_scene_out_of_range_cannot_report_ok(
    tmp_path: Path,
    fake_image_provider,
) -> None:
    """Сцена с номером вне 1..N тоже оставляет прогон без статуса `ok`."""

    state = _state(tmp_path, num_scenes=2)
    state.scenes = [_scene(1), _scene(7)]

    await ImageGeneratorStage(fake_image_provider()).run(state)

    assert state.status == "needs_review"
    assert state.metrics[-1].error is not None
