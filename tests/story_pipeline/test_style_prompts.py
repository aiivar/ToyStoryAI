"""Тесты стиль-брифа в промптах этапов (`app/story_pipeline/prompts.py`).

Проверяется не только наличие блока, но и то, что рассказчик меняет
формулировку: иначе третий голос заведомо проваливал бы требование
самопредставления героя от первого лица.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    JudgeVerdict,
    Location,
    NarrativeArc,
    PageScene,
    PageText,
    RevisionPayload,
    StoryBrief,
    StoryState,
)
from app.story_pipeline.prompts import (
    format_style_brief,
    format_style_criterion,
    format_style_guide,
)
from app.story_pipeline.stages.judge import JudgeStage
from app.story_pipeline.stages.planner import PlannerStage
from app.story_pipeline.stages.reviser import ReviserStage
from app.story_pipeline.stages.screenwriter import ScreenwriterStage
from app.story_pipeline.style import JUNIOR, SENIOR


def _brief() -> StoryBrief:
    return StoryBrief(
        logline="Плюшевый мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться и помогает зайцу.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу детской.",
            rising_action="Мишка выходит в лес и слышит плач.",
            climax="Мишка помогает зайцу у ручья.",
            resolution="Мишка и заяц возвращаются домой друзьями.",
        ),
    )


def _entities() -> ExtractedEntities:
    return ExtractedEntities(
        characters=[
            Character(
                name="Мишка",
                appearance="коричневый плюшевый медведь",
                clothing="красный шарф",
                distinguishing_features="заплатка на левом ухе",
            )
        ],
        locations=[Location(name="Лес", description="сосновый лес утром")],
        key_objects=[],
        art_style="soft watercolor storybook illustration",
    )


def _state(num_pages: int = 2) -> StoryState:
    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=num_pages,
        brief=_brief(),
        pages=[
            PageText(page_num=n, text=f"Текст страницы {n}.")
            for n in range(1, num_pages + 1)
        ],
        entities=_entities(),
        scenes=[
            PageScene(
                page_num=n,
                scene_description="Мишка в кадре.",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=[],
            )
            for n in range(1, num_pages + 1)
        ],
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_style_brief_carries_structure_and_narrator_opening() -> None:
    """Блок планировщика несёт структуру истории и голос рассказчика."""

    brief = format_style_brief(JUNIOR, "first_person")

    assert "МАНЕРА ПОВЕСТВОВАНИЯ" in brief
    assert "Структура истории" in brief
    assert "сама себя" in brief
    assert "Запреты" in brief
    assert "Не проговаривай мораль" in brief


def test_style_brief_changes_with_narrator() -> None:
    """Третий голос формулируется иначе, чем самопредставление игрушки."""

    first = format_style_brief(JUNIOR, "first_person")
    third = format_style_brief(JUNIOR, "third_person_limited")

    assert first != third
    assert "первого лица" in first
    assert "третьем лице" in third
    assert "о себе он не рассказывает" in third


def test_style_guide_carries_language_norms_of_profile() -> None:
    """Языковой блок берёт ориентиры из профиля, а не из константы."""

    junior = format_style_guide(JUNIOR, "first_person")
    senior = format_style_guide(SENIOR, "first_person")

    assert "7–10" in junior
    assert "8–10" in senior
    assert "12–19" in junior
    assert "диминутивы" in junior.lower()
    assert "принадлежат речи персонажей" in junior
    assert "по мотивам" not in junior.lower()


def test_style_criterion_is_narrow_and_narrator_aware() -> None:
    """Пятый критерий закрыт списком и меняется вместе с рассказчиком."""

    first = format_style_criterion(JUNIOR, "first_person")
    third = format_style_criterion(JUNIOR, "third_person_limited")

    assert "Манера повествования" in first
    assert "Стилистические придирки нарушением не считаются" in first
    assert "герой не представлен" in first
    assert first != third
    assert "третьем лице" in third


async def test_planner_prompt_contains_style_brief(fake_llm) -> None:
    """Этап планирования получает блок структуры и рассказчика."""

    llm = fake_llm([_brief()])

    await PlannerStage(llm).run(_state())

    user = llm.calls[0]["user"]
    assert "МАНЕРА ПОВЕСТВОВАНИЯ" in user
    assert "Структура истории" in user
    assert "первого лица" in user


async def test_screenwriter_prompt_contains_style_guide(fake_llm) -> None:
    """Этап написания получает языковой блок и запреты."""

    llm = fake_llm(
        [
            RevisionPayload(pages=[], scenes=[])
        ]
    )

    # Ответ неважен: интересует собранный запрос, поэтому падение после
    # вызова допустимо и не влияет на проверку.
    try:
        await ScreenwriterStage(llm).run(_state())
    except Exception:  # noqa: BLE001 - проверяется только промпт
        pass

    user = llm.calls[0]["user"]
    assert "МАНЕРА ПОВЕСТВОВАНИЯ" in user
    assert "фраза короткая" in user
    assert "Не спасай мир" in user


async def test_judge_prompt_contains_style_criterion(fake_llm) -> None:
    """Судья получает пятый критерий с манерой выбранного рассказчика."""

    state = _state()
    state.narrator = "third_person_limited"
    llm = fake_llm([JudgeVerdict(passed=True, issues=[], problem_pages=[])])

    await JudgeStage(llm).run(state)

    user = llm.calls[0]["user"]
    assert "по пяти критериям" in user
    assert "Манера повествования" in user
    assert "третьем лице" in user


async def test_reviser_prompt_contains_style_guide(fake_llm) -> None:
    """Ревизор сохраняет манеру: стиль-блок есть в его запросе."""

    state = _state()
    state.judge_history.append(
        JudgeVerdict(passed=False, issues=["слабая связка"], problem_pages=[1])
    )
    llm = fake_llm(
        [
            RevisionPayload(
                pages=[PageText(page_num=1, text="Новый текст страницы 1.")],
                scenes=[state.scenes[0]],
            )
        ]
    )

    await ReviserStage(llm).run(state)

    user = llm.calls[0]["user"]
    assert "МАНЕРА ПОВЕСТВОВАНИЯ" in user
    assert "фраза короткая" in user
    assert "слабая связка" in user
