"""Тесты набора иллюстрируемых страниц (`app/story_pipeline/illustrations.py`).

Правило выбора важно тем, что от него зависит платный вызов: уход набора
из диапазона означает лишние генерации или прогон без картинок. Поэтому
проверяется не только «правильно при обычных входных данных», но и то,
что вырожденные входные данные не ломают прогон.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.story_pipeline.illustrations import (
    illustrated_page_nums,
    normalize_page_nums,
    resolve_illustrated_pages,
)
from app.story_pipeline.models import StoryState


def _state(num_pages: int = 6, illustrated: list[int] | None = None) -> StoryState:
    """Состояние прогона для проверки чтения набора."""

    return StoryState(
        theme="тест",
        age=5,
        moral="мораль",
        num_pages=num_pages,
        illustrated_pages=illustrated or [],
        started_at=datetime.now(timezone.utc),
    )


# --- Вычисление набора по бюджету (D3) ----------------------------------


def test_budget_three_of_six_picks_cover_climax_and_finale() -> None:
    """Шесть страниц и три картинки — обложка, пред-финал и финал."""

    assert illustrated_page_nums(6, 3) == (1, 4, 6)


def test_zero_budget_means_every_page() -> None:
    """Нулевой бюджет отключает ограничение."""

    assert illustrated_page_nums(6, 0) == (1, 2, 3, 4, 5, 6)


def test_budget_at_or_above_page_count_means_every_page() -> None:
    """Бюджет не меньше числа страниц схлопывается в «все страницы»."""

    assert illustrated_page_nums(6, 6) == (1, 2, 3, 4, 5, 6)
    assert illustrated_page_nums(6, 20) == (1, 2, 3, 4, 5, 6)


def test_single_page_budget_picks_first_page() -> None:
    """На одну картинку остаётся обложка: без неё книжка непонятна."""

    assert illustrated_page_nums(6, 1) == (1,)


def test_single_page_book_is_illustrated_whatever_the_budget() -> None:
    """Единственная страница иллюстрируется любым бюджетом."""

    assert illustrated_page_nums(1, 3) == (1,)


@pytest.mark.parametrize(
    ("num_pages", "budget"),
    [
        (6, 3),
        (6, 2),
        (6, 4),
        (6, 5),
        (5, 3),
        (4, 3),
        (8, 3),
        (12, 5),
        (7, 6),
        (3, 2),
    ],
)
def test_computed_set_is_a_valid_nonempty_subset(num_pages: int, budget: int) -> None:
    """Набор всегда корректное непустое подмножество без повторов."""

    result = illustrated_page_nums(num_pages, budget)

    assert result, "набор иллюстрируемых страниц обязан быть непустым"
    assert len(set(result)) == len(result), "в наборе не должно быть повторов"
    assert list(result) == sorted(result), "набор должен идти по возрастанию"
    assert all(1 <= page <= num_pages for page in result)
    assert result[0] == 1, "первая страница обязана быть иллюстрируемой"
    assert result[-1] == num_pages, "последняя страница обязана быть иллюстрируемой"
    assert len(result) == min(budget, num_pages)


def test_set_is_deterministic() -> None:
    """Повторный вызов даёт тот же набор: от него зависят платные вызовы."""

    assert illustrated_page_nums(6, 3) == illustrated_page_nums(6, 3)


# --- Нормализация (D1) --------------------------------------------------


def test_normalize_keeps_valid_set_unchanged() -> None:
    """Корректный набор проходит нормализацию без изменений."""

    assert normalize_page_nums([1, 4, 6], 6) == (1, 4, 6)


def test_normalize_drops_out_of_range_numbers() -> None:
    """Номера вне диапазона страниц отбрасываются."""

    assert normalize_page_nums([1, 4, 6, 99], 6) == (1, 4, 6)
    assert normalize_page_nums([0, -3, 7], 6) == (1, 2, 3, 4, 5, 6)


def test_normalize_removes_duplicates_and_sorts() -> None:
    """Повторы снимаются, порядок приводится к возрастающему."""

    assert normalize_page_nums([6, 1, 4, 1, 6], 6) == (1, 4, 6)


def test_normalize_falls_back_to_all_pages_when_empty() -> None:
    """Пустой набор после нормализации заменяется всеми страницами.

    Иначе прогон остался бы без единой картинки и без внятной причины:
    полный набор — это безопасный откат к прежнему поведению.
    """

    assert normalize_page_nums([], 6) == (1, 2, 3, 4, 5, 6)
    assert normalize_page_nums([99], 6) == (1, 2, 3, 4, 5, 6)
    assert normalize_page_nums("не список", 6) == (1, 2, 3, 4, 5, 6)


def test_normalize_ignores_bools_and_non_ints() -> None:
    """Булевы значения и не-числа в набор не попадают.

    `True` — это `int` в Python, но набор страниц из булевых значений —
    опечатка, а не число страниц.
    """

    assert normalize_page_nums([1, True, "4", None, 6], 6) == (1, 6)


def test_normalize_accepts_any_iterable() -> None:
    """Набор может прийти генератором, а не только списком."""

    assert normalize_page_nums(iter([3, 1, 2]), 6) == (1, 2, 3)


def test_normalize_of_empty_book_is_empty() -> None:
    """Книга без страниц не превращается в выдуманный набор."""

    assert normalize_page_nums([1, 2], 0) == ()


# --- Чтение набора из состояния -----------------------------------------


def test_resolve_returns_set_from_state() -> None:
    """Заполненное поле читается как есть."""

    assert resolve_illustrated_pages(_state(6, [1, 4, 6])) == (1, 4, 6)


def test_resolve_of_empty_field_means_every_page() -> None:
    """Пустое поле — «набор не задан», то есть иллюстрируются все страницы."""

    assert resolve_illustrated_pages(_state(6, [])) == (1, 2, 3, 4, 5, 6)
    assert resolve_illustrated_pages(_state(6, None)) == (1, 2, 3, 4, 5, 6)
