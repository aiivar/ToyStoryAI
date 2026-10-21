"""Централизация промптов: текстов обращения к LLM нет вне `prompts.py`.

Требование буквальное (`specs/story-orchestration/spec.md`, критерий приёмки
№5): «все промпты к языковой модели ДОЛЖНЫ храниться в отдельном файле
`prompts.py` и не размещаться в модулях этапов или ином коде».

Проверяем это не эвристикой по внешнему виду текста (докстринги и сообщения
об ошибках — русский текст, но не промпты), а по месту использования: вне
`prompts.py` запрещены строковые литералы (в том числе f-строки), из которых
собирается обращение к модели. Это ровно те места, где промпт может быть
написан инлайн:

* получатель `.format(...)` — этап собирает промпт из шаблона;
* аргумент `complete_json(..., user=..., system=...)` — клиент получает
  готовый текст.

Домашнее правило пакета, которое делает такую проверку возможной: `.format()`
в `app/story_pipeline` используется только для шаблонов промптов; обычные
сообщения об ошибках собираются f-строками.

Проверка проходит по AST, поэтому не зависит от форматирования кода.
Дополнительно проверяется, что корректирующий промпт клиента импортирован
из `prompts`, а формулировки проблем валидатора берутся оттуда же.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.story_pipeline import prompts
from app.story_pipeline.errors import SchemaValidationError, StoryPipelineError
from app.story_pipeline.models import (
    NarrativeArc,
    PagesPayload,
    PageText,
    StoryBrief,
    StoryState,
)
from app.story_pipeline.providers import llm as llm_module
from app.story_pipeline.providers.llm import (
    Usage,
    build_schema_correction_prompt,
)
from app.story_pipeline.stages import screenwriter as screenwriter_module
from app.story_pipeline.stages.screenwriter import (
    ScreenwriterStage,
    validate_page_count,
)

PACKAGE_ROOT = Path(llm_module.__file__).resolve().parent.parent
PROMPTS_PATH = Path(prompts.__file__).resolve()

# Имена методов, аргументы которых не должны быть строковыми литералами.
FORBIDDEN_CALLS = frozenset({"format", "complete_json"})

# Имена параметров вызова LLM-клиента, передаваемые текстом.
LLM_TEXT_PARAMS = frozenset({"user", "system"})


def _story_pipeline_modules() -> list[Path]:
    """Все модули пайплайна сказок, кроме самого `prompts.py`."""

    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if path.resolve() != PROMPTS_PATH
    ]


def _prompt_literals(module: ast.Module) -> list[ast.AST]:
    """Узлы дерева, в которых наружу смотрит строковый литерал промпта."""

    found: list[ast.AST] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, "id", None
        )
        if name not in FORBIDDEN_CALLS:
            continue
        if (
            name == "format"
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Constant | ast.JoinedStr)
        ):
            # Шаблон промпта стоит перед `.format(...)`, а не среди аргументов.
            found.append(func.value)
        for argument in node.args:
            if isinstance(argument, ast.Constant | ast.JoinedStr):
                found.append(argument)
        for keyword in node.keywords:
            if keyword.arg in LLM_TEXT_PARAMS and isinstance(
                keyword.value, ast.Constant | ast.JoinedStr
            ):
                found.append(keyword.value)
    return found


def test_no_prompt_literals_outside_prompts_module() -> None:
    """Ни один модуль пайплайна, кроме `prompts.py`, не пишет текст промпта."""

    offenders: list[str] = []
    for path in _story_pipeline_modules():
        module = ast.parse(path.read_text("utf-8"))
        for node in _prompt_literals(module):
            text = ast.unparse(node)
            offenders.append(
                f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {text}"
            )

    assert offenders == [], (
        "Текст обращения к языковой модели собирается вне prompts.py:\n"
        + "\n".join(offenders)
    )


def test_prompts_module_does_hold_prompt_literals() -> None:
    """Контрольная проверка: `prompts.py` действительно наполнен шаблонами.

    Без неё предыдущий тест проходил бы и на пустом `prompts.py`.
    """

    module = ast.parse(PROMPTS_PATH.read_text("utf-8"))
    templates = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant | ast.JoinedStr)
    ]

    assert len(templates) >= 10


def test_correction_prompt_template_lives_in_prompts_module() -> None:
    """Шаблон корректирующего промпта определён в `prompts`, а не в клиенте."""

    assert (
        "Предыдущий ответ не соответствовал" in prompts.SCHEMA_CORRECTION_USER
    )
    assert "{user}" in prompts.SCHEMA_CORRECTION_USER
    assert "{error}" in prompts.SCHEMA_CORRECTION_USER
    # Клиент импортирует шаблон, а не собирает текст у себя.
    assert llm_module.SCHEMA_CORRECTION_USER is prompts.SCHEMA_CORRECTION_USER
    source = Path(llm_module.__file__).read_text("utf-8")
    assert "Предыдущий ответ не соответствовал" not in source


def test_correction_prompt_is_assembled_from_template() -> None:
    """Собранный корректирующий промпт совпадает с подстановкой в шаблон."""

    error = SchemaValidationError("Ответ LLM не соответствует схеме _Dummy")
    original = "исходный запрос этапа"

    built = build_schema_correction_prompt(original, error)

    assert built == prompts.SCHEMA_CORRECTION_USER.format(
        user=original, error=error
    )
    assert built.startswith(original)
    assert str(error) in built


def test_validator_problem_messages_come_from_prompts() -> None:
    """Формулировки проблем валидатора берутся из `prompts.py`.

    Проверяется по модулю сценариста: он импортирует шаблоны сообщений и не
    содержит их текста в своих строках.
    """

    problems = validate_page_count([PageText(page_num=1, text="Раз.")], 6)

    assert problems
    assert problems[0] == prompts.EXPECTED_PAGE_COUNT.format(
        expected=6, actual=1
    )
    source = Path(screenwriter_module.__file__).read_text("utf-8")
    assert prompts.EXPECTED_PAGE_COUNT not in source
    assert "Ожидалось ровно" not in source


def test_short_page_problem_text_lives_in_prompts_module() -> None:
    """Формулировка проблемы о короткой странице — константа `prompts.py`.

    Задача 2.2: номер страницы и её фактический объём уходят в модель, поэтому
    текст проблемы обязан жить рядом с остальными формулировками, а не в
    модуле этапа. Шаблон проверяется на подстановку, а не только наличие:
    подстановка без плейсхолдеров означала бы, что модель не получит ни
    номера страницы, ни её объёма, и перепишет всё подряд.
    """

    template = prompts.PAGE_TOO_SHORT

    assert "{page_num}" in template
    assert "{words}" in template
    assert "{expected}" in template
    rendered = template.format(page_num=4, words=31, expected=60)
    assert "4" in rendered
    assert "31" in rendered
    assert "60" in rendered
    # Текст не должен дублироваться в модуле этапа.
    source = Path(screenwriter_module.__file__).read_text("utf-8")
    assert "слов при минимуме" not in source


class _ShortPagesLLM:
    """Фейк `LLMClient`, всегда возвращающий заведомо короткие страницы.

    Отдельный мини-фейк вместо общей фикстуры: тесту нужен именно отказ по
    объёму, а не прогон целиком, поэтому заготовок на весь граф тут нет.
    """

    def __init__(self, pages: list[PageText]) -> None:
        self._pages = pages
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete_json(self, **kwargs: Any) -> tuple[Any, Usage]:
        self.calls.append(kwargs)
        return PagesPayload(pages=self._pages), Usage(
            prompt_tokens=1, completion_tokens=1
        )


async def test_short_page_problem_reaches_the_repair_request() -> None:
    """Проблема о короткой странице реально подставляется в перезапрос.

    Проверяется через публичный интерфейс этапа, а не константой: шаблон,
    который никто не подставляет, выглядел бы в тестах зелёным, а модель
    получила бы починочный запрос без указания, что именно переписать.
    """

    llm = _ShortPagesLLM(
        [
            PageText(page_num=1, text="Раз."),
            PageText(page_num=2, text="Два."),
        ]
    )
    state = StoryState(
        theme="Тема сказки",
        age=6,
        moral="Мораль сказки",
        num_pages=2,
        page_words_min=60,
        page_words_max=120,
        brief=_story_brief(),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    try:
        await ScreenwriterStage(llm).run(state)  # type: ignore[arg-type]
    except StoryPipelineError:
        # Неважно, чем закончился этап: интересует второй запрос.
        pass

    assert llm.call_count == 2
    repair_user = llm.calls[1]["user"]
    rendered = prompts.PAGE_TOO_SHORT.format(page_num=1, words=1, expected=60)
    assert rendered in repair_user
    rendered_second = prompts.PAGE_TOO_SHORT.format(
        page_num=2, words=1, expected=60
    )
    assert rendered_second in repair_user


def _story_brief() -> StoryBrief:
    """Заготовка брифа, который получает этап сценариста."""

    return StoryBrief(
        logline="Мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу.",
            rising_action="Мишка выходит в лес.",
            climax="Мишка спасает зайца.",
            resolution="Мишка и заяц дружат.",
        ),
    )


def test_scene_coverage_prompt_is_centralized() -> None:
    """Промпт починки покрытия страниц — константа `prompts.py`."""

    template = prompts.SCENE_COVERAGE_REPAIR_USER

    assert "{problems}" in template
    assert "{problem_pages}" in template
    assert "{problem_context}" in template
    assert "ровно одна сцена" in template


# --- Требование иллюстрируемости переносится на набор страниц (D5) -------


def test_illustrated_pages_plan_lives_in_prompts_module() -> None:
    """Блок про набор иллюстрируемых страниц — константа `prompts.py`.

    Проверяется тем же инвариантом централизации, что и остальные промпты:
    формулировка про визуальные вершины и связки не должна расходиться между
    шаблоном и местом подстановки.
    """

    assert "{pages}" in prompts.ILLUSTRATED_PAGES_PLAN
    assert "связк" in prompts.ILLUSTRATED_PAGES_PLAN
    assert "{illustration_plan}" in prompts.SCREENWRITER_USER
    # Прежнее требование вынесено из общего шаблона в отдельную константу:
    # при ограниченном наборе оно было бы прямой ложью.
    assert prompts.ALL_PAGES_ILLUSTRATABLE not in prompts.SCREENWRITER_USER


def test_illustration_plan_names_only_limited_set() -> None:
    """Ограниченный набор — номера и требование связок вместо «каждой»."""

    plan = prompts.format_illustration_plan([1, 4, 6], 6)

    assert "1, 4, 6" in plan
    assert "связк" in plan
    assert prompts.ALL_PAGES_ILLUSTRATABLE not in plan


def test_illustration_plan_keeps_all_pages_wording_on_full_set() -> None:
    """Полный набор — нынешняя формулировка, без перечисления номеров."""

    plan = prompts.format_illustration_plan([1, 2, 3, 4, 5, 6], 6)

    assert plan == prompts.ALL_PAGES_ILLUSTRATABLE
    assert "связк" not in plan


def test_illustration_plan_treats_unset_field_as_full_set() -> None:
    """Пустой набор — «не задан», то есть иллюстрируются все страницы."""

    assert (
        prompts.format_illustration_plan([], 3)
        == prompts.ALL_PAGES_ILLUSTRATABLE
    )
