"""Набор иллюстрируемых страниц: вычисление, нормализация и чтение.

Число страниц текста и число страниц с картинками — разные вещи. Модуль
разводит их одним типом: `illustrated_page_nums` отвечает на вопрос
«какие страницы получают картинку», а остальной код уже не вычисляет этот
набор заново.

Три функции, и каждая отвечает на свой вопрос:

- `illustrated_page_nums(num_pages, budget)` — вычислить набор из бюджета;
- `normalize_page_nums(candidate, num_pages)` — привести произвольный набор
  к корректному непустому подмножеству `1..num_pages`;
- `resolve_illustrated_pages(state)` — прочитать набор из состояния
  прогона, а если его там нет, взять все страницы.

Правило выбора (D3): крайние страницы обязательны, остальные распределены
равномерно. При шести страницах и бюджете 3 это 1, 4 и 6 — обложка,
пред-финал и финал.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - импорт только для типов
    from app.story_pipeline.models import StoryState


def illustrated_page_nums(num_pages: int, budget: int) -> tuple[int, ...]:
    """Номера иллюстрируемых страниц для числа страниц и бюджета.

    Набор всегда непуст, не содержит повторов и лежит в `1..num_pages`.
    Крайние страницы входят всегда: ребёнок узнаёт героя на первой
    странице и перелистывает книжку до конца ради развязки, поэтому
    обе должны быть с картинкой.

    Бюджет `0` и бюджет не меньше числа страниц означают одно и то же —
    иллюстрируются все страницы. Так выключатель ограничения не может
    случайно превратиться в «книжку вообще без иллюстраций».

    Позиции считаются целочисленно, без `float`: округление полувверх
    задаётся формулой `(2*num + den) // (2*den)`, иначе детерминизм
    зависел бы от представления дробных. При `1 < budget < num_pages`
    шаг между соседними позициями больше единицы, поэтому номера
    строго возрастают и дубликатов не возникает.
    """

    if num_pages < 1:
        return ()
    if budget <= 0 or budget >= num_pages:
        return tuple(range(1, num_pages + 1))
    if budget == 1:
        return (1,)

    step = budget - 1
    return tuple(
        1 + (2 * index * (num_pages - 1) + step) // (2 * step)
        for index in range(budget)
    )


def normalize_page_nums(
    candidate: object,
    num_pages: int,
) -> tuple[int, ...]:
    """Привести произвольный набор номеров к корректному виду.

    Отбрасываются номера вне `1..num_pages`, повторы снимаются, порядок
    возрастает. Если после этого ничего не осталось, возвращаются все
    страницы: пустой набор означал бы книжку, которую нечем разбирать,
    а полный — прежнее поведение, то есть безопасный откат.

    Нормализация обязательна и для явно заданного набора. Вычисленный
    набор корректен по построению, поэтому ошибаться может только тот,
    кто задал набор руками, — и без этой функции прогон падал бы позже
    и непонятно где.
    """

    if num_pages < 1:
        return ()

    numbers: set[int] = set()
    for value in _iter_ints(candidate):
        if 1 <= value <= num_pages:
            numbers.add(value)
    if not numbers:
        return tuple(range(1, num_pages + 1))
    return tuple(sorted(numbers))


def resolve_illustrated_pages(state: StoryState) -> tuple[int, ...]:
    """Набор иллюстрируемых страниц из состояния прогона.

    Пустое поле означает «набор не задан» — так состояние выглядит у
    прогонов, собранных до появления ограничения и у тех, где его
    выключили. В этом случае иллюстрируются все страницы.
    """

    if state.illustrated_pages:
        return tuple(state.illustrated_pages)
    return tuple(range(1, state.num_pages + 1))


def _iter_ints(candidate: object) -> list[int]:
    """Целые значения из того, что пришло в набор.

    Принимаются `int` и итерируемые коллекции; всё остальное считается
    пустым набором и уходит в полный набор страниц. Исключение в
    `bool` сделано намеренно: `True` — это `int` в Python, но набор
    страниц из булевых значений — опечатка, а не число страниц.
    """

    if isinstance(candidate, bool):
        return []
    if isinstance(candidate, int):
        return [candidate]
    if isinstance(candidate, str) or not hasattr(candidate, "__iter__"):
        return []
    return [value for value in candidate if isinstance(value, int) and not isinstance(value, bool)]


__all__ = [
    "illustrated_page_nums",
    "normalize_page_nums",
    "resolve_illustrated_pages",
]
