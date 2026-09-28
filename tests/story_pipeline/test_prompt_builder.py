"""Тесты чистых сборщиков промптов изображений.

`build_art_prompt` и `build_character_sheet_prompt` — чистые функции без LLM
и ввода-вывода, поэтому тесты прямые: без моков и фикстур.
"""

from __future__ import annotations

from loguru import logger

from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    KeyObject,
    Location,
    PageScene,
)
from app.story_pipeline.prompts import ART_STYLE_DEFAULT
from app.story_pipeline.stages.prompt_builder import (
    CHARACTER_SHEET_PHOTO_REFERENCE,
    build_art_prompt,
    build_character_sheet_prompt,
)

# Описание игрушки-героя: тот же источник правды, что у экстрактора и судьи.
TOY_DESCRIPTION = (
    "Зайка: маленький серый кролик, мягкая игрушка, одно ухо в пятнах."
)


def _entities() -> ExtractedEntities:
    """Небольшой реестр сущностей для проверки сборки промптов."""

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
            Location(name="Детская", description="светлая комната с окном"),
            Location(name="Лес", description="сосновый лес утром"),
            Location(name="Чердак", description="   "),
        ],
        key_objects=[
            KeyObject(name="Корзинка", description="плетёная корзинка с грибами"),
            KeyObject(name="Мяч", description=""),
            KeyObject(name="Фонарик", description="  "),
        ],
        art_style="soft watercolor storybook illustration",
    )


def test_build_art_prompt_matches_canonical_formula() -> None:
    """Промпт страницы собирается ровно по канонической формуле спецификации."""

    scene = PageScene(
        page_num=1,
        scene_description="Мишка и Заяц играют у окна",
        characters_in_scene=["Мишка", "Заяц"],
        location="Детская",
        objects_in_scene=["Корзинка"],
    )

    prompt = build_art_prompt(scene, _entities())

    assert prompt == (
        "soft watercolor storybook illustration. "
        "коричневый плюшевый медведь красный вязаный шарф заплатка на левом ухе "
        "серый пушистый заяц синие штанишки длинные уши с белыми кончиками. "
        "Мишка и Заяц играют у окна. "
        "Location: Детская (светлая комната с окном). "
        "Objects: Корзинка (плетёная корзинка с грибами). "
        "Consistent character design across all pages, "
        "children's book illustration."
    )


def test_build_art_prompt_without_characters_drops_empty_segments() -> None:
    """Пустые персонажи и объекты не оставляют двойных разделителей."""

    scene = PageScene(
        page_num=1,
        scene_description="Пустая комната утром",
        characters_in_scene=[],
        location="Детская",
        objects_in_scene=[],
    )

    prompt = build_art_prompt(scene, _entities())

    assert prompt == (
        "soft watercolor storybook illustration. "
        "Пустая комната утром. "
        "Location: Детская (светлая комната с окном). "
        "Consistent character design across all pages, "
        "children's book illustration."
    )
    assert ". ." not in prompt
    assert "  " not in prompt


def test_build_art_prompt_degrades_when_description_absent() -> None:
    """Локация/объект без описания (в том числе из пробелов) — только имя."""

    scene = PageScene(
        page_num=2,
        scene_description="На чердаке нашёлся фонарик и мяч",
        characters_in_scene=[],
        location="Чердак",
        objects_in_scene=["Фонарик", "Мяч", "Корзинка"],
    )

    prompt = build_art_prompt(scene, _entities())

    assert prompt == (
        "soft watercolor storybook illustration. "
        "На чердаке нашёлся фонарик и мяч. "
        "Location: Чердак. "
        "Objects: Фонарик, Мяч, Корзинка (плетёная корзинка с грибами). "
        "Consistent character design across all pages, "
        "children's book illustration."
    )
    assert "()" not in prompt


def test_build_art_prompt_logs_unknown_names_without_failing() -> None:
    """Неизвестные имена попадают как есть и логируются предупреждением."""

    scene = PageScene(
        page_num=3,
        scene_description="Дракон летит над космосом с мечом",
        characters_in_scene=["Дракон"],
        location="Космос",
        objects_in_scene=["Меч"],
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )

    try:
        prompt = build_art_prompt(scene, _entities())
    finally:
        logger.remove(sink_id)

    assert prompt == (
        "soft watercolor storybook illustration. "
        "Дракон. "
        "Дракон летит над космосом с мечом. "
        "Location: Космос. "
        "Objects: Меч. "
        "Consistent character design across all pages, "
        "children's book illustration."
    )
    # По одному предупреждению на неизвестные имя персонажа, локацию и объект.
    assert len(messages) == 3
    assert any("Дракон" in message for message in messages)
    assert any("Космос" in message for message in messages)
    assert any("Меч" in message for message in messages)


def test_build_art_prompt_falls_back_to_default_art_style() -> None:
    """Пустой или пробельный art_style заменяется стилем по умолчанию."""

    scene = PageScene(
        page_num=1,
        scene_description="Мишка на поляне",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )

    for blank in ("", "   "):
        entities = _entities().model_copy(update={"art_style": blank})
        prompt = build_art_prompt(scene, entities)

        assert prompt.startswith(f"{ART_STYLE_DEFAULT}. ")
        assert prompt.split(". ")[0] == ART_STYLE_DEFAULT


def test_build_character_sheet_prompt_lists_characters_and_requirements() -> None:
    """Reference-лист перечисляет персонажей с full_desc и задаёт требования."""

    entities = _entities()

    prompt = build_character_sheet_prompt(entities)

    # Каждый персонаж реестра — с дословным описанием-якорем.
    for character in entities.characters:
        assert character.full_desc in prompt
        assert character.name in prompt
    # Единый стиль сказки.
    assert entities.art_style in prompt
    # Требования reference-листа: нейтральный фон и полный рост.
    assert "Neutral plain background" in prompt
    assert "full body" in prompt
    # Детерминирована: повторный вызов даёт ту же строку.
    assert build_character_sheet_prompt(entities) == prompt


def test_build_character_sheet_prompt_carries_toy_and_reference() -> None:
    """Лист получает описание игрушки и требование опереться на фото.

    Лист генерируется с фотографией игрушки как единственным референсом (D5),
    поэтому промпт обязан и описать облик, и явно потребовать следовать фото:
    без этого модель рисует «просто плюшевого мишку» по описанию из реестра.
    """

    entities = _entities()

    prompt = build_character_sheet_prompt(entities, toy_description=TOY_DESCRIPTION)

    # Описание игрушки-героя доходит до листа дословно.
    assert TOY_DESCRIPTION in prompt
    # Требование следовать фотографии-референсу.
    assert CHARACTER_SHEET_PHOTO_REFERENCE in prompt
    # Детерминирована: повторный вызов даёт ту же строку.
    assert (
        build_character_sheet_prompt(entities, toy_description=TOY_DESCRIPTION)
        == prompt
    )


def test_build_character_sheet_prompt_without_toy_is_unchanged() -> None:
    """Без описания (CLI-сценарий) лист собирается как раньше, без мусора."""

    entities = _entities()

    prompt = build_character_sheet_prompt(entities)

    assert TOY_DESCRIPTION not in prompt
    assert CHARACTER_SHEET_PHOTO_REFERENCE not in prompt
    assert prompt == build_character_sheet_prompt(entities, toy_description="  ")
    assert ". ." not in prompt
    assert not prompt.endswith(". ")


def test_build_character_sheet_prompt_without_characters_is_clean() -> None:
    """Пустой реестр и пустой стиль не ломают лист и не плодят разделители."""

    entities = _entities().model_copy(
        update={"characters": [], "art_style": "  "}
    )

    prompt = build_character_sheet_prompt(entities)

    assert prompt.startswith(f"{ART_STYLE_DEFAULT}. ")
    assert "Neutral plain background" in prompt
    assert "full body" in prompt
    assert ". ." not in prompt
    assert "  " not in prompt
    assert not prompt.endswith(". ")
