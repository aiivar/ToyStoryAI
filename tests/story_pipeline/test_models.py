"""Тесты моделей состояния и контрактов (`app/story_pipeline/models.py`)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.story_pipeline.errors import (
    ImageGenerationError,
    LLMError,
    PermanentImageGenerationError,
    PermanentLLMError,
    SchemaValidationError,
    StoryPipelineError,
    TransientImageGenerationError,
    TransientLLMError,
)
from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    JudgeVerdict,
    KeyObject,
    Location,
    NarrativeArc,
    PageArt,
    PageScene,
    PageText,
    Stage,
    StageMetrics,
    StoryBrief,
    StoryState,
)


def _full_state() -> StoryState:
    """Собрать заполненное состояние со всеми артефактами."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=5,
        moral="Смелость помогает найти друзей",
        num_pages=2,
        brief=StoryBrief(
            logline="Мишка идёт через лес и находит друга.",
            controlling_idea="Дружба начинается со смелости.",
            character_arc="Мишка учится не бояться.",
            narrative_arc=NarrativeArc(
                exposition="Мишка живёт в шкафу.",
                rising_action="Мишка выходит в лес.",
                climax="Мишка помогает зайцу.",
                resolution="Они становятся друзьями.",
            ),
        ),
        pages=[
            PageText(page_num=1, text="Жил-был мишка."),
            PageText(page_num=2, text="Мишка нашёл друга."),
        ],
        entities=ExtractedEntities(
            characters=[
                Character(
                    name="Мишка",
                    appearance="коричневый плюшевый медведь",
                    clothing="красный шарф",
                    distinguishing_features="заплатка на ухе",
                )
            ],
            locations=[Location(name="Лес", description="сосновый лес утром")],
            key_objects=[KeyObject(name="Корзинка", description="плетёная корзинка")],
            art_style="soft watercolor storybook illustration",
        ),
        scenes=[
            PageScene(
                page_num=1,
                scene_description="Мишка в шкафу",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=[],
            ),
            PageScene(
                page_num=2,
                scene_description="Мишка и заяц",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=["Корзинка"],
            ),
        ],
        art_prompts=[(1, "prompt 1"), (2, "prompt 2")],
        character_sheet_path="output/run/character_sheet.png",
        character_sheet_seed=42,
        images=[
            PageArt(page_num=1, path="output/run/page_1.png", seed=1),
            PageArt(page_num=2, path="output/run/page_2.png", seed=None),
        ],
        judge_history=[
            JudgeVerdict(passed=False, issues=["слабая связка"], problem_pages=[2]),
            JudgeVerdict(passed=True, issues=[]),
        ],
        revision_count=1,
        status="ok",
        unresolved_pages=[2],
        metrics=[
            StageMetrics(
                stage="plan",
                duration_ms=12.5,
                attempts=1,
                prompt_tokens=100,
                completion_tokens=50,
            )
        ],
        output_dir="output/run",
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
    )


def test_full_desc_join_order() -> None:
    """full_desc склеивает три поля через пробел в фиксированном порядке."""

    character = Character(
        name="Мишка",
        appearance="коричневый плюшевый медведь",
        clothing="красный шарф",
        distinguishing_features="заплатка на ухе",
    )

    assert (
        character.full_desc
        == "коричневый плюшевый медведь красный шарф заплатка на ухе"
    )


def test_full_desc_is_computed_not_stored() -> None:
    """full_desc — вычисляемое поле, а не часть схемы сериализации."""

    character = Character(
        name="Мишка",
        appearance="a",
        clothing="b",
        distinguishing_features="c",
    )

    assert "full_desc" not in character.model_dump()
    assert "full_desc" not in Character.model_json_schema()["properties"]


def test_state_round_trip_json() -> None:
    """dump(mode="json") → model_validate восстанавливает состояние."""

    state = _full_state()

    dumped = state.model_dump(mode="json")
    # Дамп должен быть полностью JSON-сериализуемым (бинарей нет).
    json.dumps(dumped)
    restored = StoryState.model_validate(dumped)

    assert restored == state
    assert restored.character_sheet_path == "output/run/character_sheet.png"
    assert restored.images[1].seed is None
    assert restored.started_at == state.started_at


def test_dump_has_no_binary_and_keeps_character_sheet() -> None:
    """Сериализация содержит путь и seed character sheet, без байтов."""

    state = _full_state()
    dumped = state.model_dump(mode="json")

    assert dumped["character_sheet_path"] == "output/run/character_sheet.png"
    assert dumped["character_sheet_seed"] == 42
    assert all(isinstance(image["path"], str) for image in dumped["images"])
    assert isinstance(dumped["started_at"], str)


def test_illustrated_pages_round_trip_and_default_is_isolated() -> None:
    """Набор иллюстрируемых страниц переживает JSON-цикл.

    Набор лежит в `story.json` рядом со списком изображений, поэтому по
    артефакту прогона видно, какие страницы не проиллюстрированы по
    замыслу, а какие — из-за сбоя генерации. Без сериализации этого
    различения в артефакте не осталось бы вовсе.
    """

    state = _full_state().model_copy(update={"illustrated_pages": [1, 4, 6]})

    dumped = state.model_dump(mode="json")
    restored = StoryState.model_validate(dumped)

    assert restored.illustrated_pages == [1, 4, 6]
    assert restored == state


def test_illustrated_pages_default_is_empty_and_isolated() -> None:
    """Пустой набор по умолчанию: «набор не задан», а не «рисовать ничего»."""

    assert StoryState(
        theme="т", age=4, moral="м", started_at=datetime.now(timezone.utc)
    ).illustrated_pages == []
    # Списки по умолчанию не должны разделяться между экземплярами.
    first = StoryState(
        theme="т", age=4, moral="м", started_at=datetime.now(timezone.utc)
    )
    first.illustrated_pages.append(1)
    second = StoryState(
        theme="т", age=4, moral="м", started_at=datetime.now(timezone.utc)
    )
    assert second.illustrated_pages == []


def test_page_words_round_trip_with_short_pages() -> None:
    """Границы объёма и недобор переживают JSON-цикл.

    По `story.json` должно быть видно, какой объём просили и какие страницы
    не дотянули: без этого разбор прогона не отличил бы «модель не
    попала в диапазон» от «объём не измерялся вовсе».
    """

    state = _full_state().model_copy(
        update={"page_words_min": 80, "page_words_max": 140, "short_pages": [2]}
    )

    dumped = state.model_dump(mode="json")
    json.dumps(dumped)
    restored = StoryState.model_validate(dumped)

    assert restored.page_words_min == 80
    assert restored.page_words_max == 140
    assert restored.short_pages == [2]
    assert restored == state


def test_page_words_default_to_unset_and_short_pages_to_empty() -> None:
    """Не заданные границы — `None`, недобор — пустой список.

    `None`, а не 60/120: иначе нельзя отличить «границы не заданы»
    (их возьмёт оркестратор из настроек) от явно заданного числа.
    """

    def make_state() -> StoryState:
        return StoryState(
            theme="t",
            age=4,
            moral="m",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    state = make_state()

    assert state.page_words_min is None
    assert state.page_words_max is None
    assert state.short_pages == []
    # Мутабельный список по умолчанию не делится между экземплярами.
    state.short_pages.append(1)
    assert make_state().short_pages == []


def test_short_pages_survive_round_trip_of_minimal_state() -> None:
    """Даже минимальное состояние с недобором остаётся разбираемым."""

    state = StoryState(
        theme="t",
        age=4,
        moral="m",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ).model_copy(update={"page_words_min": 60, "page_words_max": 120})

    restored = StoryState.model_validate(state.model_dump(mode="json"))

    assert restored.short_pages == []
    assert restored.page_words_max == 120


def test_toy_description_round_trips_as_plain_text() -> None:
    """Описание игрушки переживает JSON-цикл и остаётся строкой, не бинарём."""

    state = _full_state().model_copy(
        update={"toy_description": "Коричневый плюшевый медвежонок в шарфе."}
    )

    dumped = state.model_dump(mode="json")
    json.dumps(dumped)
    restored = StoryState.model_validate(dumped)

    assert restored.toy_description == state.toy_description
    assert isinstance(dumped["toy_description"], str)


def test_toy_description_defaults_to_none() -> None:
    """Без фото (CLI-сценарий) описание игрушки пусто, а не выдумано."""

    assert _full_state().toy_description is None


def test_judge_verdict_defaults_are_isolated() -> None:
    """Мутабельный default problem_pages не разделяется между инстансами."""

    first = JudgeVerdict(passed=True, issues=[])
    second = JudgeVerdict(passed=True, issues=[])

    first.problem_pages.append(1)

    assert first.problem_pages == [1]
    assert second.problem_pages == []


def test_unresolved_pages_default_is_isolated() -> None:
    """Мутабельный default unresolved_pages не разделяется между инстансами."""

    def make_state() -> StoryState:
        return StoryState(
            theme="t",
            age=4,
            moral="m",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    first = make_state()
    second = make_state()

    first.unresolved_pages.append(1)

    assert first.unresolved_pages == [1]
    assert second.unresolved_pages == []


def test_story_state_status_default_and_literal() -> None:
    """Статус по умолчанию — ok; недопустимое значение отклоняется."""

    state = StoryState(
        theme="t",
        age=4,
        moral="m",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert state.status == "ok"
    assert state.num_pages == 6
    assert state.revision_count == 0

    with pytest.raises(ValidationError):
        StoryState(
            theme="t",
            age=4,
            moral="m",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            status="unknown",  # type: ignore[arg-type]
        )


def test_stage_protocol_is_structural() -> None:
    """Объект с name и async run соответствует протоколу Stage."""

    class DummyStage:
        name = "dummy"

        async def run(self, state: StoryState) -> StoryState:
            return state

    stage: Stage = DummyStage()

    assert isinstance(stage, Stage)
    assert stage.name == "dummy"


def test_error_taxonomy() -> None:
    """Иерархия ошибок: временные/постоянные подклассы базовых ошибок."""

    assert isinstance(TransientLLMError("x"), LLMError)
    assert isinstance(PermanentLLMError("x"), LLMError)
    assert isinstance(SchemaValidationError("x"), LLMError)
    assert isinstance(LLMError("x"), StoryPipelineError)
    assert isinstance(TransientImageGenerationError("x"), ImageGenerationError)
    assert isinstance(PermanentImageGenerationError("x"), ImageGenerationError)
    assert isinstance(ImageGenerationError("x"), StoryPipelineError)
    # Текстовые и графические ошибки — разные ветви дерева.
    assert not issubclass(ImageGenerationError, LLMError)
    assert not issubclass(LLMError, ImageGenerationError)
