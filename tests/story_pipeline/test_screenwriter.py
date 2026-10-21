"""Тесты сценариста страниц и валидации числа страниц.

Этап проверяется через публичный интерфейс `async def run(state)` с фейковым
`LLMClient`; чистая функция `validate_page_count` — напрямую, без моков.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from loguru import logger

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
    validate_page_length,
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


def _state(num_pages: int = 6, illustrated: list[int] | None = None) -> StoryState:
    """Собрать состояние с готовым брифом для этапа сценариста."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=num_pages,
        illustrated_pages=illustrated or [],
        brief=_brief(),
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def _pages(nums: list[int]) -> PagesPayload:
    """Ответ сценариста: страницы нужного объёма с заданными номерами.

    Текст добивается до дефолтных 60 слов не случайно, а чтобы тесты про
    число страниц и промпты не спотыкались о проверку объёма: дефолтные
    границы применяются ко всем прогонам, и двухсловная страница в заготовке
    означала бы лишний перезапрос вместо проверяемого поведения.
    """

    return PagesPayload(
        pages=[
            PageText(page_num=n, text=f"Текст страницы {n}. {_filler(70)}")
            for n in nums
        ]
    )


def _filler(total: int) -> str:
    """Хвост из `total` слов для доведения страницы до нужного объёма."""

    return " ".join(["слово"] * total)


async def test_screenwriter_stores_pages_sorted_by_number(fake_llm) -> None:
    """Ровно N страниц из ответа кладутся в `state.pages` по порядку номеров."""

    llm = fake_llm([_pages([3, 1, 5, 2, 6, 4])])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)

    result = await stage.run(state)

    assert result is state
    assert [page.page_num for page in state.pages] == [1, 2, 3, 4, 5, 6]
    assert state.pages[0].text.startswith("Текст страницы 1.")
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
    # Требования к содержанию страниц. Объём задан словами, а не предложениями:
    # предложения модель считает неточно, и такой заказ нечем проверить.
    assert "60–120 слов" in user
    assert "сенсорн" in user
    # Пустое поле набора = «иллюстрируются все страницы», поэтому в промпте
    # остаётся требование иллюстрируемости каждой страницы.
    assert "иллюстрируем" in user
    assert "доступную ребёнку 6 лет" in user
    # Схема из модели, а не руками.
    assert "$defs" in user


# --- Набор иллюстрируемых страниц в промпте (D5) -------------------------


async def test_prompt_names_illustrated_pages_and_calls_others_transitions(
    fake_llm,
) -> None:
    """Ограниченный набор: номера названы, остальные страницы — связки.

    Требование иллюстрируемости переносится с «каждой страницы» на
    конкретные номера: визуальные вершины должны совпасть с нарисованными
    страницами, иначе книжка рисуется не там, где драма.
    """

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state(6, illustrated=[1, 4, 6]))

    user = llm.calls[0]["user"]
    assert "1, 4, 6" in user
    assert "связк" in user
    # Прежняя формулировка при ограниченном наборе стала бы ложью: модель
    # старалась бы сделать визуальной каждую из шести страниц.
    assert "Каждая страница должна быть иллюстрируема" not in user


async def test_prompt_keeps_all_pages_illustratable_on_full_set(fake_llm) -> None:
    """При полном наборе остаётся нынешнее требование ко всем страницам."""

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state(6, illustrated=[1, 2, 3, 4, 5, 6]))

    user = llm.calls[0]["user"]
    assert "Каждая страница должна быть иллюстрируема" in user
    # Перечисления номеров при полном наборе не нужно: оно только шумит.
    assert "связк" not in user


async def test_prompt_without_set_in_state_asks_for_all_pages(fake_llm) -> None:
    """Пустое поле — «набор не задан», то есть иллюстрируются все страницы."""

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])
    state = _state(6)
    assert state.illustrated_pages == []

    await ScreenwriterStage(llm).run(state)

    assert "Каждая страница должна быть иллюстрируема" in llm.calls[0]["user"]


async def test_repair_request_carries_the_same_illustrated_set(fake_llm) -> None:
    """Починочный запрос собран из `base_user`, поэтому набор в нём есть.

    Иначе исправленный текст потерял бы требование к визуальной сцене: набор
    в первой попытке — единственное место, где модель о нём узнаёт.
    """

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state(6, illustrated=[1, 4, 6]))

    assert llm.call_count == 2
    repair_user = llm.calls[1]["user"]
    assert "1, 4, 6" in repair_user
    assert "связк" in repair_user
    assert "Каждая страница должна быть иллюстрируема" not in repair_user


async def test_repair_request_keeps_full_set_wording(fake_llm) -> None:
    """При полном наборе починочный запрос говорит то же, что и исходный."""

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state(6, illustrated=[1, 2, 3, 4, 5, 6]))

    assert "Каждая страница должна быть иллюстрируема" in llm.calls[1]["user"]


# --- Границы объёма страницы в промптах ---------------------------------


def _state_with_bounds(
    num_pages: int = 6,
    words_min: int = 60,
    words_max: int = 120,
    illustrated: list[int] | None = None,
) -> StoryState:
    """Состояние с разрешёнными границами объёма — как их оставил пайплайн."""

    state = _state(num_pages, illustrated)
    state.page_words_min = words_min
    state.page_words_max = words_max
    return state


async def test_prompt_carries_run_word_bounds(fake_llm) -> None:
    """Промпт сценариста задаёт объём страницы числами этого прогона.

    Требование «2–3 предложения» было неисполнимо: модель считает предложения
    неточно, и на живых прогонах выдавала 27–41 слово вместо задуманного.
    Число слов, наоборот, можно посчитать кодом и проверить.
    """

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state_with_bounds(6, 70, 150))

    user = llm.calls[0]["user"]
    assert "70" in user
    assert "150" in user
    # Прежнее требование снято: вместе с новым оно противоречило бы друг другу.
    assert "2–3 предложения" not in user


async def test_prompt_uses_default_bounds_when_state_has_none(fake_llm) -> None:
    """Без разрешённых границ в промпт уходит дефолт, а не пустое место.

    Прогоны, собирающие состояние сами (тесты, CLI без оркестратора), не
    проходят через разрешение границ, поэтому дефолтный запас обязателен.
    """

    llm = fake_llm([_pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(_state())

    user = llm.calls[0]["user"]
    assert "60" in user
    assert "120" in user


async def test_repair_request_carries_both_bounds_and_illustrated_set(
    fake_llm,
) -> None:
    """Починочный запрос несёт и набор страниц, и границы объёма.

    `SCREENWRITER_REPAIR_USER` вставляет `base_user` целиком, поэтому обе
    подстановки доезжают без отдельного копирования. Если починочный запрос
    потеряет границы, модель перепишет недобор обратно в краткую форму —
    то есть вторая попытка испортит то, что исправляла.
    """

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 2, 3, 4, 5, 6])])

    await ScreenwriterStage(llm).run(
        _state_with_bounds(6, 70, 150, illustrated=[1, 4, 6])
    )

    repair_user = llm.calls[1]["user"]
    assert "1, 4, 6" in repair_user
    assert "связк" in repair_user
    assert "70" in repair_user
    assert "150" in repair_user


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


# --- Чистая функция validate_page_length --------------------------------


def _words_page(page_num: int, count: int) -> PageText:
    """Страница с текстом ровно из `count` слов."""

    return PageText(page_num=page_num, text=" ".join(["слово"] * count))


def test_validate_page_length_accepts_pages_within_bounds() -> None:
    """Текст в границах (включая ровно минимум) — проблем нет."""

    pages = [_words_page(1, 60), _words_page(2, 90), _words_page(3, 120)]

    assert validate_page_length(pages, 60) == []


def test_validate_page_length_reports_page_below_minimum() -> None:
    """Короткая страница даёт проблему с номером и фактическим числом слов.

    Число слов уходит в модель как предмет адресной правки: без него починочный
    запрос не объясняет, что именно и насколько разворачивать.
    """

    pages = [_words_page(1, 60), _words_page(2, 31), _words_page(3, 60)]

    problems = validate_page_length(pages, 60)

    assert len(problems) == 1
    assert "2" in problems[0]
    assert "31" in problems[0]
    assert "60" in problems[0]


def test_validate_page_length_counts_words_by_split() -> None:
    """Счёт слов — `len(text.split())`, как его считает проверка.

    Значимость: порог, заданный моделью «на глаз», несовместим с порогом кода,
    и тогда проверка ловит не недобор, а особенности разбивки текста.
    """

    page = PageText(page_num=1, text="  раз   два\nтри  ")

    problems = validate_page_length([page], 3)

    assert problems == []
    assert validate_page_length([page], 4)


def test_validate_page_length_ignores_overlong_pages() -> None:
    """Длиннее максимума — не проблема: проверяется нижняя граница.

    Резать текст сверху было бы тем же, от чего изменение отказывается:
    верхняя граница нужна модели как ориентир, а не как стоп-сигнал.
    """

    assert validate_page_length([_words_page(1, 200)], 60) == []


def test_validate_page_length_reports_every_short_page() -> None:
    """Недобор виден по каждой странице, а не по первой попавшейся."""

    pages = [_words_page(1, 10), _words_page(2, 20), _words_page(3, 5)]

    problems = validate_page_length(pages, 60)

    assert len(problems) == 3
    assert any("1" in problem for problem in problems)
    assert any("10" in problem for problem in problems)


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


# --- Разные последствия двух видов неуспеха -----------------------------


def _long_pages(nums: list[int]) -> PagesPayload:
    """Ответ сценариста со страницами нужного объёма."""

    return PagesPayload(
        pages=[
            PageText(page_num=n, text=" ".join(["слово"] * 70))
            for n in nums
        ]
    )


async def test_short_page_triggers_exactly_one_repair(fake_llm) -> None:
    """Недобор объёма уходит в починочный запрос и лечится одним повтором."""

    # Первая страница первой попытки короткая, число страниц верное.
    first = PagesPayload(
        pages=[
            PageText(page_num=1, text="Коротко."),
            _long_pages([2]).pages[0],
            _long_pages([3]).pages[0],
        ]
    )
    llm = fake_llm([first, _long_pages([1, 2, 3])])
    state = _state_with_bounds(3)

    await ScreenwriterStage(llm).run(state)

    assert llm.call_count == 2
    assert [page.page_num for page in state.pages] == [1, 2, 3]
    # Повтор вылечил: деградации нет.
    assert state.short_pages == []
    metric = state.metrics[-1]
    assert metric.attempts == 2
    assert metric.error is None


async def test_short_page_after_repair_degrades_without_raising(
    fake_llm,
) -> None:
    """Недобор после перезапроса не роняет прогон, а виден в артефакте.

    Разница с числом страниц принципиальна: неверное число ломает книжку
    структурно — читать её нечем, а короткая страница косметика. Родитель
    получает шесть страниц текста, пять из которых нужного объёма, поэтому
    отнимать у него всю книжку из-за недостающих слов на странице 4 —
    цена, несоразмерная проблеме.
    """

    short = PagesPayload(
        pages=[
            PageText(page_num=1, text="Очень коротко."),
            _long_pages([2]).pages[0],
            _long_pages([3]).pages[0],
        ]
    )
    llm = fake_llm([short, short])
    state = _state_with_bounds(3)

    result = await ScreenwriterStage(llm).run(state)

    # Прогон не прерван и книжка доставлена как есть.
    assert result is state
    assert [page.page_num for page in state.pages] == [1, 2, 3]
    assert llm.call_count == 2
    # Деградация видна в артефакте прогона и в метрике этапа.
    assert state.short_pages == [1]
    metric = state.metrics[-1]
    assert metric.stage == "script"
    assert metric.error is not None
    assert "1" in metric.error


async def test_short_pages_warning_is_logged(fake_llm) -> None:
    """Причина деградации попадает в лог: разбор прогона её не пропустит."""

    short = PagesPayload(
        pages=[
            PageText(page_num=1, text="Очень коротко."),
            _long_pages([2]).pages[0],
        ]
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )
    try:
        await ScreenwriterStage(fake_llm([short, short])).run(
            _state_with_bounds(2)
        )
    finally:
        logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "1" in joined
    assert "60" in joined


async def test_wrong_page_count_still_raises_after_repair(fake_llm) -> None:
    """Неверное число страниц по-прежнему роняет прогон.

    Поведение не менялось: структурная поломка после перезапроса означает, что
    читать книжку нечем, и отдавать её «как есть» нельзя.
    """

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 3, 5])])
    state = _state_with_bounds(6)

    with pytest.raises(PipelineValidationError):
        await ScreenwriterStage(llm).run(state)

    assert llm.call_count == 2
    assert state.short_pages == []


async def test_short_pages_are_not_confused_with_numbering_failure(
    fake_llm,
) -> None:
    """После неуспеха по числу страниц номера недобора не записываются.

    Прогон упал — состояние не «доехало» до книжки, и `short_pages` в артефакте
    был бы враньём: страниц, которые читатель получит, не существует.
    """

    llm = fake_llm([_pages([1, 2, 3]), _pages([1, 3, 5])])
    state = _state_with_bounds(6)

    with pytest.raises(PipelineValidationError):
        await ScreenwriterStage(llm).run(state)

    assert state.short_pages == []


async def test_repair_response_with_wrong_count_does_not_mask_short_pages(
    fake_llm,
) -> None:
    """Ошибка о числе страниц главнее недобора и не прячется за ним."""

    # Первая попытка короткая по объёму, вторая — уже сломана структурно.
    short = PagesPayload(
        pages=[
            PageText(page_num=1, text="Коротко."),
            _long_pages([2]).pages[0],
            _long_pages([3]).pages[0],
        ]
    )
    llm = fake_llm([short, _pages([1, 2])])
    state = _state_with_bounds(3)

    with pytest.raises(PipelineValidationError):
        await ScreenwriterStage(llm).run(state)

    assert state.short_pages == []


async def test_screenwriter_requires_brief(fake_llm) -> None:
    """Сценарист без брифа падает типизированной ошибкой, не обращаясь к LLM."""

    llm = fake_llm([])
    stage = ScreenwriterStage(llm)
    state = _state(num_pages=6)
    state.brief = None

    with pytest.raises(StoryPipelineError):
        await stage.run(state)

    assert llm.call_count == 0
