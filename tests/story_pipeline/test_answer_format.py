"""Строгая форма ответа LLM: инструкция генерируется из pydantic-модели.

Контекст бага (живой прогон): когда промпт заканчивается сырой JSON-схемой,
модель копирует СЛУЖЕБНЫЕ ключи самой схемы (`description`, `properties`,
`required`, `type`, `title`) как форму ответа. Судья вернул
`{"description": …, "properties": {…}}` вместо
`{"passed": …, "issues": […]}`.

Лечится системно: инструкция о форме ответа собирается из той же
pydantic-модели, которую рендерит `render_schema`, и подставляется в конец
каждого промпта, который просит JSON. Ручных списков полей нет — они
выводятся из модели, поэтому инструкция не может разойтись с контрактом.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel

from app.story_pipeline.models import (
    ExtractedEntities,
    JudgeVerdict,
    Location,
    NarrativeArc,
    PagesPayload,
    PageScene,
    PageText,
    RevisionPayload,
    ScenesPayload,
    StoryBrief,
    StoryState,
)
from app.story_pipeline.prompts import (
    ANSWER_EXAMPLE_LABEL,
    SCHEMA_META_KEYS,
    model_meta_keys,
    render_answer_format,
)
from app.story_pipeline.stages.entity_extractor import EntityExtractorStage
from app.story_pipeline.stages.judge import JudgeStage
from app.story_pipeline.stages.planner import PlannerStage
from app.story_pipeline.stages.reviser import ReviserStage
from app.story_pipeline.stages.scene_breakdown import SceneBreakdownStage
from app.story_pipeline.stages.screenwriter import ScreenwriterStage

# Модели всех восьми этапов, которые запрашивают JSON у LLM.
STAGE_RESPONSE_MODELS = (
    StoryBrief,
    PagesPayload,
    ExtractedEntities,
    ScenesPayload,
    JudgeVerdict,
    RevisionPayload,
)


def _example_from(text: str) -> dict[str, Any]:
    """Достать JSON-заготовку ответа из строки формата."""

    marker = f"{ANSWER_EXAMPLE_LABEL}"
    tail = text[text.index(marker) + len(marker) :]
    start = tail.index("{")
    end = tail.rindex("}") + 1
    parsed: dict[str, Any] = json.loads(tail[start:end])
    return parsed


def test_answer_format_lists_top_level_fields_of_the_model() -> None:
    """Инструкция перечисляет верхнеуровневые поля нужной схемы."""

    rendered = render_answer_format(JudgeVerdict)

    assert "passed" in rendered
    assert "issues" in rendered
    assert "problem_pages" in rendered


def test_answer_format_forbids_json_schema_meta_keys() -> None:
    """Служебные ключи JSON-схемы запрещены — именно их копировала модель."""

    rendered = render_answer_format(JudgeVerdict)

    for key in SCHEMA_META_KEYS:
        assert key in rendered
    assert "НЕЛЬЗЯ" in rendered


def test_stage_models_do_not_collide_with_schema_meta_keys() -> None:
    """Ни одна модель этапа не объявляет поле, совпадающее со служебным ключом.

    Пока это так, запрет в инструкции не спорит с перечислением полей. Строка
    одна, но она держит весь инвариант: добавление в модель поля `title`
    обязано валить её, иначе промпт сам себе противоречит.
    """

    collisions = {
        model.__name__: sorted(set(SCHEMA_META_KEYS) & set(model.model_fields))
        for model in STAGE_RESPONSE_MODELS
    }

    assert {name: keys for name, keys in collisions.items() if keys} == {}


def test_model_meta_keys_drop_colliding_field() -> None:
    """Совпадающее поле выпадает из запрета, а не конфликтует с ним.

    Синтетическая модель повторяет тот баг, который инструкция чинит: если
    бы в модели было поле `description`, промпт велел бы вернуть ключ
    `description` и тут же запрещал его.
    """

    class Colliding(BaseModel):
        description: str
        title: str

    rendered = render_answer_format(Colliding)

    assert "верни JSON-объект РОВНО" in rendered
    # Поле осталось в перечислении, а запрета на него больше нет.
    assert "description" in rendered.split(ANSWER_EXAMPLE_LABEL)[0]
    assert "`description`" not in rendered
    assert "`title`" not in rendered
    assert "НЕЛЬЗЯ" in rendered  # запрет остался — но не на эти поля
    # Остальные служебные ключи, не совпавшие с полями, запрещены по-прежнему.
    for key in ("properties", "required", "type"):
        assert f"`{key}`" in rendered
    assert model_meta_keys(Colliding) == ("properties", "required", "type")


def test_answer_format_without_meta_rule_keeps_the_text_readable() -> None:
    """Без запрещаемых ключей в шаблоне не остаётся дыр и висячих строк."""

    class AllColliding(BaseModel):
        description: str
        properties: str
        required: str
        type: str
        title: str

    rendered = render_answer_format(AllColliding)

    assert "НЕЛЬЗЯ" not in rendered
    assert ";;" not in rendered
    assert rendered.count("\n\n") == 0
    assert ANSWER_EXAMPLE_LABEL in rendered


@pytest.mark.parametrize("model", STAGE_RESPONSE_MODELS, ids=lambda m: m.__name__)
def test_answer_example_validates_against_its_own_model(model: type) -> None:
    """Пример-заготовка честен: модель принимает его как свой ответ.

    Это гарантирует и что генератор работает, и что пример не врёт про форму
    контракта: если в модель добавят поле, перестанет хватать значения
    генератора, и тест упадёт.
    """

    example = _example_from(render_answer_format(model))

    assert model.model_validate(example) is not None


@pytest.mark.parametrize("model", STAGE_RESPONSE_MODELS, ids=lambda m: m.__name__)
def test_answer_example_has_exactly_the_model_fields(model: type) -> None:
    """В примере ровно те ключи, что объявлены в модели, и все они нужны."""

    example = _example_from(render_answer_format(model))

    assert set(example) == set(model.model_fields)


def test_answer_example_of_page_wrapper_shows_nested_page_shape() -> None:
    """Список страниц в примере показывает форму вложенного объекта.

    Пустой список был бы местом, куда модель кладёт «заготовку» вместо
    содержимого, поэтому вложенный объект в примере присутствует.
    """

    example = _example_from(render_answer_format(PagesPayload))

    assert example["pages"] == [
        PageText(page_num=1, text="...").model_dump()
    ]


# --- Инструкция доходит до каждого этапа, который просит JSON ---------------
#
# Проверяется не константа, а реально отправленный `user`: только так видно,
# что этап не забыл подставить инструкцию. Судья — тот этап, где баг и был
# найден, остальные семь пока просто везли.


def _brief() -> StoryBrief:
    return StoryBrief(
        logline="Мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться и помогает зайцу.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу детской.",
            rising_action="Мишка выходит в лес и слышит плач.",
            climax="Мишка спасает зайца от ручья.",
            resolution="Мишка и Заяц возвращаются домой друзьями.",
        ),
    )


def _entities() -> ExtractedEntities:
    return ExtractedEntities(
        characters=[],
        locations=[Location(name="Лес", description="сосновый лес")],
        key_objects=[],
        art_style="soft watercolor storybook illustration",
    )


def _pages(nums: list[int]) -> list[PageText]:
    # Текст добивается до 70 слов: дефолтные границы объёма страницы
    # применяются к каждому прогону, и короткая заготовка означала бы лишний
    # корректирующий перезапрос вместо проверяемого поведения.
    return [
        PageText(
            page_num=n, text=f"Текст страницы {n}. " + " ".join(["слово"] * 70)
        )
        for n in nums
    ]


def _scene(page_num: int, name: str = "нет") -> PageScene:
    return PageScene(
        page_num=page_num,
        scene_description=f"Кадр страницы {page_num}.",
        characters_in_scene=[name] if name != "нет" else [],
        location="Лес",
        objects_in_scene=[],
    )


def _state() -> StoryState:
    """Состояние, на котором запускаются все этапы теста."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=5,
        moral="Смелость помогает найти друзей",
        num_pages=2,
        brief=_brief(),
        pages=_pages([1, 2]),
        entities=_entities(),
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def test_every_json_requesting_template_carries_the_format_block() -> None:
    """Ни один шаблон, отдающий `{schema}`, не забывает про форму ответа.

    Страховка от повторения бага в новом этапе: пока инструкция приходит
    отдельной подстановкой, её легко не добавить. Проверка идёт по самим
    шаблонам `prompts.py`, поэтому ловит пропуск до того, как этап начнёт
    спрашивать живую модель.
    """

    from app.story_pipeline import prompts

    templates = {
        name: value
        for name, value in vars(prompts).items()
        if isinstance(value, str) and "{schema}" in value
    }

    assert templates, "в prompts.py не осталось шаблонов со схемой"
    missing = sorted(
        name
        for name, template in templates.items()
        if "{answer_format}" not in template
    )

    assert missing == [], f"шаблоны без формы ответа: {missing}"


def _assert_strict_format(prompt: str, model: type) -> None:
    """Промпт этапа содержит имена полей схемы и запрет служебных ключей."""

    for field in model.model_fields:
        assert field in prompt, f"{model.__name__}: нет поля {field}"
    for key in SCHEMA_META_KEYS:
        assert f"`{key}`" in prompt, f"{model.__name__}: нет запрета {key}"
    assert ANSWER_EXAMPLE_LABEL in prompt


async def test_planner_prompt_constrains_answer_shape(fake_llm) -> None:
    """Планировщик получает форму ответа `StoryBrief`."""

    llm = fake_llm([_brief()])

    await PlannerStage(llm).run(_state())

    _assert_strict_format(llm.calls[0]["user"], StoryBrief)


async def test_screenwriter_prompt_constrains_answer_shape(fake_llm) -> None:
    """Сценарист получает форму ответа `PagesPayload`."""

    llm = fake_llm([PagesPayload(pages=_pages([1, 2]))])

    await ScreenwriterStage(llm).run(_state())

    _assert_strict_format(llm.calls[0]["user"], PagesPayload)


async def test_screenwriter_repair_prompt_keeps_answer_shape(fake_llm) -> None:
    """Ремонт числа страниц не теряет форму ответа.

    Ремонтный промпт вставляет в себя базовый запрос, поэтому инструкция
    приходит вместе с ним. Проверяется именно это: без проверки ремонт стал
    бы единственным путём, где модель снова видит сырую схему без формы.
    """

    llm = fake_llm(
        [PagesPayload(pages=_pages([1])), PagesPayload(pages=_pages([1, 2]))]
    )

    await ScreenwriterStage(llm).run(_state())

    assert llm.call_count == 2
    _assert_strict_format(llm.calls[1]["user"], PagesPayload)


async def test_extractor_prompt_constrains_answer_shape(fake_llm) -> None:
    """Экстрактор сущностей получает форму ответа `ExtractedEntities`."""

    llm = fake_llm([_entities()])

    await EntityExtractorStage(llm).run(_state())

    _assert_strict_format(llm.calls[0]["user"], ExtractedEntities)


async def test_scene_prompt_constrains_answer_shape(fake_llm) -> None:
    """Раскадровщик получает форму ответа `ScenesPayload`."""

    llm = fake_llm([ScenesPayload(scenes=[_scene(1), _scene(2)])])

    await SceneBreakdownStage(llm).run(_state())

    _assert_strict_format(llm.calls[0]["user"], ScenesPayload)


async def test_judge_prompt_constrains_answer_shape(fake_llm) -> None:
    """Судья получает форму ответа `JudgeVerdict` — здесь был найден баг."""

    llm = fake_llm([JudgeVerdict(passed=True, issues=[])])

    await JudgeStage(llm).run(_state())

    _assert_strict_format(llm.calls[0]["user"], JudgeVerdict)


async def test_reviser_prompt_constrains_answer_shape(fake_llm) -> None:
    """Ревизор получает форму ответа `RevisionPayload`."""

    llm = fake_llm([RevisionPayload(pages=[], scenes=[])])
    state = _state()
    state.judge_history.append(
        JudgeVerdict(passed=False, issues=["Страница 1 пустая."])
    )
    state.scenes = [_scene(1), _scene(2)]

    await ReviserStage(llm).run(state)

    _assert_strict_format(llm.calls[0]["user"], RevisionPayload)


async def test_scene_repair_prompt_constrains_answer_shape(fake_llm) -> None:
    """Ремонт ссылок раскадровки тоже несёт форму ответа.

    Ремонтный промпт — самый склонный к копированию: он идёт после неудачи,
    и модель охотно возвращает «описание ошибки» вместо сцен.
    """

    state = _state()
    bad = ScenesPayload(
        scenes=[
            _scene(1, "Мишка"),
            _scene(2),
        ]
    )
    fixed = ScenesPayload(scenes=[_scene(1), _scene(2)])
    llm = fake_llm([bad, fixed])

    await SceneBreakdownStage(llm).run(state)

    assert llm.call_count == 2
    _assert_strict_format(llm.calls[1]["user"], ScenesPayload)


async def test_coverage_repair_prompt_constrains_answer_shape(fake_llm) -> None:
    """Ремонт покрытия страниц тоже несёт форму ответа."""

    state = _state()
    incomplete = ScenesPayload(scenes=[_scene(1)])
    repaired = ScenesPayload(scenes=[_scene(1), _scene(2)])
    llm = fake_llm([incomplete, repaired])

    await SceneBreakdownStage(llm).run(state)

    assert llm.call_count == 2
    _assert_strict_format(llm.calls[1]["user"], ScenesPayload)


async def test_schema_correction_prompt_constrains_answer_shape(
    fake_llm,
) -> None:
    """Корректирующий промпт клиента сохраняет форму ответа.

    Живой прогон показал, что единственный корректирующий повтор не спас
    разорванную форму: он переспрашивал ту же схему. Поэтому исходная
    инструкция о форме должна пережить вставку в него.
    """

    from app.story_pipeline.errors import SchemaValidationError
    from app.story_pipeline.prompts import (
        JUDGE_USER,
        format_arc_plan,
        format_optional_entity_registry,
        format_scene_pages,
        format_toy_description,
    )

    base = JUDGE_USER.format(
        logline="логлайн",
        controlling_idea="мысль",
        arc_plan=format_arc_plan(_brief().narrative_arc, 2),
        age=5,
        toy_description=format_toy_description(None),
        entities=format_optional_entity_registry(None),
        story=format_scene_pages([], []),
        schema="{}",
        answer_format=render_answer_format(JudgeVerdict),
    )
    error = SchemaValidationError("Ответ LLM не соответствует схеме")

    from app.story_pipeline.providers.llm import (
        build_schema_correction_prompt,
    )

    corrected = build_schema_correction_prompt(base, error)

    _assert_strict_format(corrected, JudgeVerdict)
