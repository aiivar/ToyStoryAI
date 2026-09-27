"""Тесты сценариста страниц и валидации числа страниц.

Этап проверяется через публичный интерфейс `async def run(state)` с фейковым
`LLMClient`; чистая функция `validate_page_count` — напрямую, без моков.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.story_pipeline.errors import (
    PipelineValidationError,
    StoryPipelineError,
)
from app.story_pipeline.models import (
    NarrativeArc,
    PagesPayload,
    PageText,
    StoryBrief,
    StoryState,
)
from app.story_pipeline.stages.screenwriter import (
    ScreenwriterStage,
    validate_page_count,
)


def _brief() -> StoryBrief:
    """Заготовка брифа, которую «получает» сценарист."""

    return StoryBrief(
        logline="Плюшевый мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться и помогает зайцу.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу детской.",
            rising_action="Мишка выходит в лес и слышит плач.",
            climax="Мишка спасает зайца от ручья.",
            resolution="Мишка и заяц возвращаются домой друзьями.",
        ),
    )


def _state(num_pages: int = 6) -> StoryState:
    """Собрать состояние с готовым брифом для этапа сценариста."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=num_pages,
        brief=_brief(),
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def _pages(nums: list[int]) -> PagesPayload:
    """Ответ сценариста: страницы с заданными номерами."""

    return PagesPayload(
        pages=[
            PageText(page_num=n, text=f"Текст страницы {n}.")
            for n in nums
        ]
    )


async def test_screenwriter_stores_pages_sorted_by_number(fake_llm) -> None:
    """Ровно N страниц из ответа кладутся в `state.pages` по порядку номеров."""

    llm = fake_llm([_pages([3, 1, 5, 2, 6, 4])])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)

    result = await stage.run(state)

    assert result is state
    assert [page.page_num for page in state.pages] == [1, 2, 3, 4, 5, 6]
    assert state.pages[0].text == "Текст страницы 1."
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is PagesPayload
    assert llm.calls[0]["temperature"] == 0.8


async def test_screenwriter_prompt_binds_pages_to_arc(fake_llm) -> None:
    """Промпт несёт число страниц, биты арки с диапазонами и требования к тексту."""

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert "ровно из 6 страниц" in user
    # Биты арки из брифа дословно.
    assert "Мишка живёт в шкафу детской." in user
    assert "Мишка выходит в лес и слышит плач." in user
    assert "Мишка спасает зайца от ручья." in user
    assert "Мишка и заяц возвращаются домой друзьями." in user
    # Явная привязка страниц к битам: диапазоны для 6 страниц.
    assert "1–2" in user
    assert "3–4" in user
    # Требования к содержанию страниц.
    assert "2–3 предложения" in user
    assert "сенсорн" in user
    assert "иллюстрируем" in user
    assert "доступную ребёнку 6 лет" in user
    # Схема из модели, а не руками.
    assert "$defs" in user


# --- Чистая функция validate_page_count ---------------------------------


def _page(page_num: int) -> PageText:
    """Страница с заданным номером для проверок валидатора."""

    return PageText(page_num=page_num, text=f"Текст {page_num}.")


def test_validate_page_count_accepts_exact_number() -> None:
    """Ровно N страниц с нумерацией 1..N — проблем нет."""

    pages = [_page(n) for n in (1, 2, 3, 4, 5, 6)]

    assert validate_page_count(pages, 6) == []


def test_validate_page_count_reports_fewer_pages() -> None:
    """Меньше N страниц — проблема о неверном числе."""

    pages = [_page(1), _page(2), _page(3)]

    problems = validate_page_count(pages, 6)

    assert problems
    assert any("6" in problem and "3" in problem for problem in problems)


def test_validate_page_count_reports_gap_in_numbering() -> None:
    """Разрыв нумерации при верном числе страниц — проблема об пропуске."""

    pages = [_page(n) for n in (1, 2, 3, 4, 5, 7)]

    problems = validate_page_count(pages, 6)

    assert problems
    assert any("6" in problem for problem in problems)


def test_validate_page_count_reports_duplicates() -> None:
    """Повторяющийся номер страницы — проблема о дубле."""

    pages = [_page(n) for n in (1, 2, 3, 3, 4, 5)]

    problems = validate_page_count(pages, 6)

    assert any("дубл" in problem.lower() for problem in problems)
    assert any(
        "дубл" in problem.lower() and "3" in problem for problem in problems
    )


def test_validate_page_count_reports_non_one_based() -> None:
    """Номер 0 (не 1-based) — проблема о нумерации с 1.

    `PageText.page_num` требует `ge=1`, поэтому такой ответ отклоняется ещё
    схемой; функция обязана ловить нарушение и на всякий случай.
    """

    pages = [
        PageText.model_construct(page_num=0, text="Нулевая страница.")
    ] + [_page(n) for n in (1, 2, 3, 4, 5)]

    problems = validate_page_count(pages, 6)

    assert any("начинаться с 1" in problem for problem in problems)


# --- Корректирующий перезапрос сценариста -------------------------------


async def test_screenwriter_repairs_wrong_page_count_once(fake_llm) -> None:
    """Неверное число страниц вызывает ровно один перезапрос и правку."""

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 2, 3, 4, 5, 6])])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)

    await stage.run(state)

    assert [page.page_num for page in state.pages] == [1, 2, 3, 4, 5, 6]
    assert llm.call_count == 2
    # Перезапрос идёт с тем же system и несёт описание проблем исходного ответа.
    assert llm.calls[1]["system"] == llm.calls[0]["system"]
    repair_user = llm.calls[1]["user"]
    assert llm.calls[0]["user"] in repair_user
    assert "6" in repair_user
    assert "3" in repair_user

    metric = state.metrics[-1]
    assert metric.attempts == 2
    # Токены обеих попыток суммируются (11 + 11, 22 + 22).
    assert metric.prompt_tokens == 22
    assert metric.completion_tokens == 44


async def test_screenwriter_raises_error_when_repair_fails(fake_llm) -> None:
    """Если перезапрос не исправил число/нумерацию — типизированная ошибка."""

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 3, 5])])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)

    with pytest.raises(PipelineValidationError):
        await stage.run(state)

    assert llm.call_count == 2
    # Состояние не «спасается» молча: страницы не записаны.
    assert state.pages == []


async def test_screenwriter_requires_brief(fake_llm) -> None:
    """Сценарист без брифа падает типизированной ошибкой, не обращаясь к LLM."""

    llm = fake_llm([])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)
    state.brief = None

    with pytest.raises(StoryPipelineError):
        await stage.run(state)

    assert llm.call_count == 0
