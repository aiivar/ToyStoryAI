"""Тесты судьи, нормализации вердикта, ревизора и петли ревизии.

Этапы проверяются через публичный интерфейс `async def run(state)` с фейковым
`LLMClient`; чистые функции — напрямую, без моков.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.story_pipeline.errors import (
    PipelineValidationError,
    StoryPipelineError,
)
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
from app.story_pipeline.stages.judge import JudgeStage, validate_judge_verdict

# Описание игрушки-героя в прогоне: тот же источник правды, что у экстрактора.
TOY_DESCRIPTION = (
    "Плюшевый медвежонок: коричневый, мягкий плюш, заплатка на левом ухе."
)
from app.story_pipeline.stages.quality_loop import QualityReviewLoop
from app.story_pipeline.stages.reviser import ReviserStage



def _brief() -> StoryBrief:
    """Бриф, который «видит» судья как замысел сказки."""

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


def _entities() -> ExtractedEntities:
    """Небольшой реестр сущностей для перепроверки ссылок сцен."""

    return ExtractedEntities(
        characters=[
            Character(
                name="Мишка",
                appearance="коричневый плюшевый медведь",
                clothing="красный вязаный шарф",
                distinguishing_features="заплатка на левом ухе",
            ),
            Character(
                name="Заяц",
                appearance="серый пушистый заяц",
                clothing="синие штанишки",
                distinguishing_features="длинные уши с белыми кончиками",
            ),
        ],
        locations=[
            Location(name="Детская", description="светлая комната"),
            Location(name="Лес", description="сосновый лес утром"),
        ],
        key_objects=[],
        art_style="soft watercolor storybook illustration",
    )


def _pages(nums: list[int]) -> list[PageText]:
    """Страницы с заданными номерами."""

    return [PageText(page_num=n, text=f"Текст страницы {n}.") for n in nums]


def _scene(page_num: int) -> PageScene:
    """Корректная сцена страницы со ссылками строго на реестр."""

    return PageScene(
        page_num=page_num,
        scene_description=f"Мишка в кадре на странице {page_num}.",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )


def _state(num_pages: int = 6) -> StoryState:
    """Состояние с готовыми страницами и сценами для этапов качества."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=num_pages,
        brief=_brief(),
        pages=_pages(list(range(1, num_pages + 1))),
        entities=_entities(),
        scenes=[_scene(n) for n in range(1, num_pages + 1)],
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


# --- Этап судьи ---------------------------------------------------------


async def test_judge_records_verdict_and_metric(fake_llm) -> None:
    """Вердикт судьи пишется в историю, а метрика — в общий список."""

    verdict = JudgeVerdict(
        passed=True, issues=[], problem_pages=[]
    )
    llm = fake_llm([verdict])
    stage = JudgeStage(llm)
    state = _state()

    result = await stage.run(state)

    assert result is state
    assert state.judge_history == [verdict]
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is JudgeVerdict
    # Судья выносит вердикт строго с нулевой температурой.
    assert llm.calls[0]["temperature"] == 0.0
    metric = state.metrics[-1]
    assert metric.stage == "judge"
    assert metric.attempts == 1
    assert metric.prompt_tokens == 11
    assert metric.completion_tokens == 22


async def test_judge_prompt_covers_three_criteria_and_brief(
    fake_llm,
) -> None:
    """Промпт несёт три критерия, контекст брифа, возраст и пары «текст+сцена»."""

    llm = fake_llm([JudgeVerdict(passed=True, issues=[], problem_pages=[])])
    stage = JudgeStage(llm)
    state = _state()

    await stage.run(state)

    user = llm.calls[0]["user"]
    # Критерий 1: соответствие текста и сцены.
    assert "Соответствие текста и сцены" in user
    # Критерий 2: нарративная целостность с явными битами.
    assert "Нарративная целостность" in user
    assert "завязка" in user
    assert "кульминация" in user
    assert "развязка" in user
    # Критерий 3: возрастная безопасность под целевой возраст.
    assert "Возрастная безопасность" in user
    assert "6 лет" in user
    # Требование заполнять проблемные страницы.
    assert "problem_pages" in user
    # Контекст брифа из состояния.
    assert "Плюшевый мишка идёт через лес и находит друга." in user
    assert "Смелость открывает дорогу к дружбе." in user
    assert "Мишка спасает зайца от ручья." in user
    # Каждая страница идёт вместе со своей сценой.
    assert "Текст страницы 1." in user
    assert "Мишка в кадре на странице 1." in user
    assert "Текст страницы 6." in user
    # Схема берётся из модели, а не пишется руками.
    assert '"title": "JudgeVerdict"' in user


async def test_judge_prompt_lists_four_criteria(fake_llm) -> None:
    """Промпт судьи содержит четыре критерия: три прежних и новый.

    Критерий согласованности описания героя добавлен (D13) и не вытеснил ни
    одного прежнего: без текст-сцена, целостности и безопасности ревизия
    теряет то, ради чего существует.
    """

    llm = fake_llm([JudgeVerdict(passed=True, issues=[], problem_pages=[])])
    stage = JudgeStage(llm)

    await stage.run(_state())

    user = llm.calls[0]["user"]
    # Прежние три критерия на месте, дословно.
    assert "Соответствие текста и сцены" in user
    assert "Нарративная целостность" in user
    assert "Возрастная безопасность" in user
    # Четвёртый: согласованность описания героя между текстом и реестром.
    assert "Согласованность описания героя" in user
    assert "не противоречит" in user
    # Критериев ровно четыре — заголовок не устарел.
    assert "по четырём критериям" in user


async def test_judge_prompt_receives_toy_description_and_registry(
    fake_llm,
) -> None:
    """Судье нужны описание игрушки и реестр, иначе критерий невыполним.

    Реестр в промпте судьи раньше отсутствовал: четвёртый критерий требовал
    сверить текст с визуальным описанием, которого модель не видела, — такой
    «проверяемый» критерий всегда проходил бы вслепую.
    """

    llm = fake_llm([JudgeVerdict(passed=True, issues=[], problem_pages=[])])
    stage = JudgeStage(llm)
    state = _state()
    state.toy_description = TOY_DESCRIPTION

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert TOY_DESCRIPTION in user
    for character in _entities().characters:
        assert character.full_desc in user
    assert _entities().art_style in user
    # Сводка планировщика в промпт судьи не подставляется.
    assert state.theme not in user


async def test_judge_prompt_without_toy_description_is_clean(fake_llm) -> None:
    """Пустое описание и пустой реестр не роняют судью и не дают «None»."""

    llm = fake_llm([JudgeVerdict(passed=True, issues=[], problem_pages=[])])
    stage = JudgeStage(llm)
    state = _state()
    state.toy_description = None
    state.entities = None

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert "Описание игрушки-героя" in user
    assert "Реестр сущностей" in user
    assert "None" not in user
    assert "\n\n\n" not in user


async def test_judge_requires_brief(fake_llm) -> None:
    """Судья без брифа падает типизированно, не обращаясь к LLM."""

    llm = fake_llm([])
    stage = JudgeStage(llm)
    state = _state()
    state.brief = None

    with pytest.raises(StoryPipelineError):
        await stage.run(state)

    assert llm.call_count == 0


# --- Чистая функция validate_judge_verdict ------------------------------


def _page_numbers() -> list[PageText]:
    """Страницы 1..6 для проверок диапазона проблемных номеров."""

    return _pages([1, 2, 3, 4, 5, 6])


def test_validate_judge_verdict_drops_out_of_range_and_duplicates() -> None:
    """Номера вне диапазона отбрасываются, дубликаты схлопываются."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["Сцена не совпадает с текстом."],
        problem_pages=[0, 2, 2, 7, 3],
    )

    normalized = validate_judge_verdict(verdict, _page_numbers())

    assert normalized.problem_pages == [2, 3]
    # Остальные поля вердикта сохраняются.
    assert normalized.passed is False
    assert normalized.issues == ["Сцена не совпадает с текстом."]


def test_validate_judge_verdict_fills_all_pages_when_failed_and_empty() -> None:
    """Не пройденный вердикт без номеров консервативно считает все страницы."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["История распадается."],
        problem_pages=[],
    )

    normalized = validate_judge_verdict(verdict, _page_numbers())

    assert normalized.problem_pages == [1, 2, 3, 4, 5, 6]


def test_validate_judge_verdict_keeps_passed_verdict_unchanged() -> None:
    """Пройденный вердикт не нормализуется к «все страницы»."""

    verdict = JudgeVerdict(passed=True, issues=[], problem_pages=[])

    normalized = validate_judge_verdict(verdict, _page_numbers())

    assert normalized.passed is True
    assert normalized.problem_pages == []

    # Вне диапазона по-прежнему отбрасывается, но список не расширяется.
    partial = JudgeVerdict(passed=True, issues=[], problem_pages=[2, 9])
    assert validate_judge_verdict(partial, _page_numbers()).problem_pages == [2]


# --- Этап ревизора ------------------------------------------------------


async def test_reviser_applies_only_problem_pages(fake_llm) -> None:
    """Правки применяются только к проблемным страницам, остальные не меняются."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["Сцена страницы 3 не соответствует тексту."],
        problem_pages=[3],
    )
    revised_page = PageText(page_num=3, text="Новый текст страницы 3.")
    revised_scene = PageScene(
        page_num=3,
        scene_description="Мишка у ручья.",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [RevisionPayload(pages=[revised_page], scenes=[revised_scene])]
    )
    stage = ReviserStage(llm)
    state = _state()
    state.judge_history.append(verdict)
    before_pages = {page.page_num: page.text for page in state.pages}
    before_scenes = {
        scene.page_num: scene.scene_description for scene in state.scenes
    }

    result = await stage.run(state)

    assert result is state
    assert state.pages[2].text == "Новый текст страницы 3."
    assert state.scenes[2].scene_description == "Мишка у ручья."
    # Непроблемные страницы и сцены не тронуты.
    for number in (1, 2, 4, 5, 6):
        assert state.pages[number - 1].text == before_pages[number]
        assert state.scenes[number - 1].scene_description == before_scenes[number]
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is RevisionPayload
    assert llm.calls[0]["temperature"] == 0.7
    metric = state.metrics[-1]
    assert metric.stage == "reviser"
    assert metric.prompt_tokens == 11
    assert metric.completion_tokens == 22


async def test_reviser_prompt_carries_issues_and_problem_context(
    fake_llm,
) -> None:
    """Промпт несёт замечания судьи, контекст брифа и данные проблемной страницы."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["Страница 2 пугает ребёнка."],
        problem_pages=[2],
    )
    revised_page = PageText(page_num=2, text="Мягкий текст страницы 2.")
    revised_scene = PageScene(
        page_num=2,
        scene_description="Мишка спокойно идёт по лесу.",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [RevisionPayload(pages=[revised_page], scenes=[revised_scene])]
    )
    stage = ReviserStage(llm)
    state = _state()
    state.judge_history.append(verdict)

    await stage.run(state)

    user = llm.calls[0]["user"]
    # Замечание судьи ушло в промпт дословно.
    assert "Страница 2 пугает ребёнка." in user
    # Контекст брифа.
    assert "Смелость открывает дорогу к дружбе." in user
    # Проблемная страница: текущий текст и текущая сцена.
    assert "Текст страницы 2." in user
    assert "Мишка в кадре на странице 2." in user
    # Непроблемные страницы в промпт не подмешиваются.
    assert "Текст страницы 1." not in user
    assert "Мишка в кадре на странице 1." not in user
    # Схема ревизора из модели.
    assert '"title": "RevisionPayload"' in user


async def test_reviser_rechecks_scene_refs_after_revision(fake_llm) -> None:
    """После правки ссылки сцен проверяются заново, остаток помечается."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["Сцена страницы 3 не совпадает с текстом."],
        problem_pages=[3],
    )
    # Модель сослалась на неизвестное имя и на имя в другом регистре.
    revised_scene = PageScene(
        page_num=3,
        scene_description="Дракон и мишка у ручья.",
        characters_in_scene=["Дракон", "мишка"],
        location="Лес",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [
            RevisionPayload(
                pages=[PageText(page_num=3, text="Новый текст страницы 3.")],
                scenes=[revised_scene],
            )
        ]
    )
    stage = ReviserStage(llm)
    state = _state()
    state.judge_history.append(verdict)

    await stage.run(state)

    # Имя в другом регистре нормализовано, неизвестное — не потеряно.
    assert state.scenes[2].characters_in_scene == ["Дракон", "Мишка"]
    # Остаток неизвестных имён помечен как проблемная страница.
    assert state.unresolved_pages == [3]


async def test_reviser_requires_judge_verdict(fake_llm) -> None:
    """Без вердикта судьи ревизор падает типизированно и не зовёт LLM."""

    llm = fake_llm([])
    stage = ReviserStage(llm)
    state = _state()

    with pytest.raises(StoryPipelineError):
        await stage.run(state)

    assert llm.call_count == 0


async def test_reviser_targets_unresolved_pages_when_judge_passed(
    fake_llm,
) -> None:
    """Страницы из `unresolved_pages` правятся, даже если судья пропустил сказку."""

    verdict = JudgeVerdict(passed=True, issues=[], problem_pages=[])
    revised_page = PageText(page_num=4, text="Новый текст страницы 4.")
    revised_scene = PageScene(
        page_num=4,
        scene_description="Мишка у озера.",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [RevisionPayload(pages=[revised_page], scenes=[revised_scene])]
    )
    stage = ReviserStage(llm)
    state = _state()
    state.judge_history.append(verdict)
    state.unresolved_pages = [4]

    await stage.run(state)

    assert llm.call_count == 1
    user = llm.calls[0]["user"]
    assert "Текст страницы 4." in user
    assert "Мишка в кадре на странице 4." in user
    assert state.pages[3].text == "Новый текст страницы 4."
    # После правки проблема снята, пометка очищена.
    assert state.unresolved_pages == []


async def test_reviser_rejects_text_edit_without_scene(fake_llm) -> None:
    """Правка текста без пересобранной сцены — нарушение контракта ревизии."""

    verdict = JudgeVerdict(
        passed=False,
        issues=["Текст страницы 3 непонятен ребёнку."],
        problem_pages=[3],
    )
    llm = fake_llm(
        [
            RevisionPayload(
                pages=[PageText(page_num=3, text="Совсем другой текст.")],
                scenes=[],
            )
        ]
    )
    stage = ReviserStage(llm)
    state = _state()
    state.judge_history.append(verdict)

    with pytest.raises(PipelineValidationError):
        await stage.run(state)

    assert llm.call_count == 1
    # Состояние не «спасается» молча: текст проблемной страницы не изменён.
    assert state.pages[2].text == "Текст страницы 3."


# --- Петля ревизии ------------------------------------------------------


def _revision(page_num: int, *, tag: str) -> RevisionPayload:
    """Ответ ревизора: правка одной страницы вместе с её сценой."""

    return RevisionPayload(
        pages=[PageText(page_num=page_num, text=f"Правка {tag} страницы.")],
        scenes=[
            PageScene(
                page_num=page_num,
                scene_description=f"Правка {tag} сцены.",
                characters_in_scene=["Мишка"],
                location="Лес",
                objects_in_scene=[],
            )
        ],
    )


async def test_loop_passes_without_revision(fake_llm) -> None:
    """Пройденный с первой попытки вердикт не запускает ревизора."""

    judge_llm = fake_llm(
        [JudgeVerdict(passed=True, issues=[], problem_pages=[])]
    )
    reviser_llm = fake_llm([])
    loop = QualityReviewLoop(JudgeStage(judge_llm), ReviserStage(reviser_llm))
    state = _state()

    result = await loop.run(state)

    assert result is state
    assert state.revision_count == 0
    assert len(state.judge_history) == 1
    assert judge_llm.call_count == 1
    assert reviser_llm.call_count == 0
    assert state.status == "ok"


async def test_loop_revises_until_pass(fake_llm) -> None:
    """Непройденная проверка запускает ревизию, после правки судья пропускает."""

    fail = JudgeVerdict(
        passed=False,
        issues=["Сцена страницы 3 не совпадает с текстом."],
        problem_pages=[3],
    )
    ok = JudgeVerdict(passed=True, issues=[], problem_pages=[])
    judge_llm = fake_llm([fail, ok])
    reviser_llm = fake_llm([_revision(3, tag="первой")])
    loop = QualityReviewLoop(JudgeStage(judge_llm), ReviserStage(reviser_llm))
    state = _state()

    await loop.run(state)

    assert state.revision_count == 1
    assert [verdict.passed for verdict in state.judge_history] == [False, True]
    assert judge_llm.call_count == 2
    assert reviser_llm.call_count == 1
    assert state.status == "ok"
    # Правка применена и к тексту, и к сцене.
    assert state.pages[2].text == "Правка первой страницы."
    assert state.scenes[2].scene_description == "Правка первой сцены."


async def test_loop_marks_needs_review_after_budget(fake_llm) -> None:
    """После двух ревизий без успеха прогон сохраняется со статусом needs_review."""

    def fail() -> JudgeVerdict:
        return JudgeVerdict(
            passed=False,
            issues=["История распадается."],
            problem_pages=[3],
        )

    judge_llm = fake_llm([fail(), fail(), fail()])
    reviser_llm = fake_llm(
        [_revision(3, tag="первой"), _revision(3, tag="второй")]
    )
    loop = QualityReviewLoop(JudgeStage(judge_llm), ReviserStage(reviser_llm))
    state = _state()

    result = await loop.run(state)

    assert result is state
    # Ровно две ревизии, затем остановка без исключения.
    assert state.revision_count == 2
    assert judge_llm.call_count == 3
    assert reviser_llm.call_count == 2
    assert len(state.judge_history) == 3
    assert state.status == "needs_review"
    # Результат сохранён: страницы на месте.
    assert [page.page_num for page in state.pages] == [1, 2, 3, 4, 5, 6]


async def test_loop_revises_unresolved_pages_despite_passed_verdict(
    fake_llm,
) -> None:
    """Пометка раскадровки о неизвестных ссылках запускает ревизию даже при passed."""

    ok = JudgeVerdict(passed=True, issues=[], problem_pages=[])
    judge_llm = fake_llm([ok, ok])
    reviser_llm = fake_llm([_revision(2, tag="ссылочной")])
    loop = QualityReviewLoop(JudgeStage(judge_llm), ReviserStage(reviser_llm))
    state = _state()
    # Раскадровка пометила страницу 2: сцена ссылается на неизвестное имя.
    state.scenes[1] = PageScene(
        page_num=2,
        scene_description="Дракон у леса.",
        characters_in_scene=["Дракон"],
        location="Лес",
        objects_in_scene=[],
    )
    state.unresolved_pages = [2]

    await loop.run(state)

    assert judge_llm.call_count == 2
    assert reviser_llm.call_count == 1
    assert state.revision_count == 1
    # Правка сняла неизвестную ссылку, пометка очищена.
    assert state.scenes[1].characters_in_scene == ["Мишка"]
    assert state.unresolved_pages == []
    assert state.status == "ok"
