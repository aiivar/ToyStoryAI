"""Чистые сборщики промптов изображений: без языковой модели и без I/O.

Здесь живёт детерминированная сборка строк для генератора картинок:
промпт страницы (`build_art_prompt`) и промпт reference-листа персонажей
(`build_character_sheet_prompt`). В отличие от промптов к LLM (они собраны
в `prompts.py`), это промпты изображений, поэтому литеральные куски лежат
рядом с собирающими их чистыми функциями: формула из спецификации
`story-illustration` видна в одном месте со своим кодом.
"""

from __future__ import annotations

from loguru import logger

from app.story_pipeline.models import ExtractedEntities, PageScene
from app.story_pipeline.prompts import ART_STYLE_DEFAULT

# Требование согласованности дизайна персонажей между страницами и стиля
# детской книжной иллюстрации — дословно из спецификации
# `story-illustration`. Без завершающей точки: точку добавляет общий
# разделитель сегментов, иначе перед следующим требованием получается `. .`.
CONSISTENCY_TAIL = (
    "Consistent character design across all pages, "
    "children's book illustration"
)

# Требование отсутствия текста на картинке. Страница иллюстрирует сцену,
# а подпись рисует модель случайно и почти всегда с нечитаемыми
# псевдобукквами — такой артефакт родителю показывать нельзя.
NO_TEXT_TAIL = (
    "No text, no letters, no numbers, no captions, no speech bubbles, "
    "no signature, no watermark."
)

# Разделитель непустых сегментов промпта.
SEGMENT_SEPARATOR = ". "

# --- Промпт reference-листа персонажей (character sheet) ----------------

# Заголовок и требование полного роста: лист нужен как эталон позы и
# пропорций, поэтому каждый персонаж показан от макушки до пят.
CHARACTER_SHEET_HEADER = (
    "Character reference sheet, full body, head to toe, front view"
)

# Требование нейтрального фона: фон листа не должен конкурировать с
# описанием персонажей и попадать в сгенерированные страницы.
CHARACTER_SHEET_BACKGROUND = "Neutral plain background, no scenery or props"

# Разделитель перечня персонажей: имя — дословное описание-якорь.
CHARACTER_SHEET_CHARACTER_SEPARATOR = "; "

# Завершающая фраза листа: он задаёт согласованный дизайн для всех страниц.
CHARACTER_SHEET_TAIL = "Consistent character design for all pages of the book."

# Требование следовать фотографии-референсу. Лист генерируется с фото
# игрушки как единственным reference (D5), поэтому промпт говорит об этом
# прямо: иначе модель рисует «просто плюшевого мишку» по описанию из
# реестра, и фото перестаёт быть источником правды.
CHARACTER_SHEET_PHOTO_REFERENCE = (
    "Match the character to the provided reference photo of the toy"
)


def _join_segments(segments: tuple[str, ...]) -> str:
    """Соединить непустые сегменты разделителем, отбросив пустые.

    Пустые (и состоящие из пробелов) сегменты не оставляют лишних
    разделителей вида `. .` или двойных пробелов: результат остаётся
    детерминированным и аккуратным при любом наборе непустых частей.
    """

    cleaned = [segment.strip() for segment in segments if segment.strip()]
    return SEGMENT_SEPARATOR.join(cleaned)


def _resolve_art_style(entities: ExtractedEntities) -> str:
    """Вернуть стиль сказки, подставив стиль по умолчанию при пустом.

    Пробельная строка считается пустой: иначе сегмент стиля исчез бы из
    промпта, и страницы потеряли бы единый стиль.
    """

    return entities.art_style.strip() or ART_STYLE_DEFAULT


def _warn_unknown(kind: str, name: str) -> None:
    """Предупредить о ссылке сцены на имя, которого нет в реестре.

    Сборка промпта не падает: имя подставляется как есть, а прогон виден в
    логах для последующей адресной правки раскадровки.
    """

    logger.warning(
        "art prompt: unknown {} name '{}' is not in the registry; "
        "using the raw name",
        kind,
        name,
    )


def build_art_prompt(scene: PageScene, entities: ExtractedEntities) -> str:
    """Собрать промпт изображения страницы по канонической формуле.

    Чистая функция: ни языковой модели, ни ввода-вывода. Описания
    персонажей подставляются дословно (`Character.full_desc`) как якорь
    внешности; локация и объекты обогащаются описаниями из реестра.
    Неизвестные имена не роняют сборку: они попадают в промпт как есть.
    """

    characters_by_name = {
        character.name: character.full_desc
        for character in entities.characters
    }
    character_descriptions: list[str] = []
    for name in scene.characters_in_scene:
        if name in characters_by_name:
            character_descriptions.append(characters_by_name[name])
        else:
            _warn_unknown("character", name)
            character_descriptions.append(name)
    characters = " ".join(character_descriptions)

    location_descriptions = {
        location.name: location.description for location in entities.locations
    }
    if scene.location and scene.location not in location_descriptions:
        _warn_unknown("location", scene.location)
    location_description = (
        location_descriptions.get(scene.location, "").strip()
    )
    location = (
        f"{scene.location} ({location_description})"
        if location_description
        else scene.location
    )
    location_segment = f"Location: {location}" if location.strip() else ""

    object_descriptions = {
        item.name: item.description for item in entities.key_objects
    }
    object_parts: list[str] = []
    for name in scene.objects_in_scene:
        if name not in object_descriptions:
            _warn_unknown("object", name)
            object_parts.append(name)
            continue
        description = object_descriptions[name].strip()
        object_parts.append(f"{name} ({description})" if description else name)
    objects = ", ".join(object_parts)
    objects_segment = f"Objects: {objects}" if objects.strip() else ""

    return _join_segments(
        (
            _resolve_art_style(entities),
            characters,
            scene.scene_description,
            location_segment,
            objects_segment,
            CONSISTENCY_TAIL,
            NO_TEXT_TAIL,
        )
    )


def _toy_reference_segment(toy_description: str | None) -> str:
    """Собрать блок «описание игрушки + требование следовать фото».

    Обе части уходят в промпт только вместе: описание без требования модель
    легко трактует как пожелание, а требование без описания — как пустое
    указание. Пустое или пробельное описание (CLI-сценарий без фото) убирает
    обе части, чтобы лист не требовал несуществующего reference.
    """

    cleaned = (toy_description or "").strip()
    if not cleaned:
        return ""
    return f"{cleaned}. {CHARACTER_SHEET_PHOTO_REFERENCE}"


def build_character_sheet_prompt(
    entities: ExtractedEntities,
    toy_description: str | None = None,
) -> str:
    """Собрать промпт reference-листа персонажей (character sheet).

    Лист генерируется один раз до страниц и передаётся им как reference.
    Перечисляет всех персонажей реестра с дословным `full_desc`, требует
    нейтральный фон и полный рост и использует единый стиль сказки.

    `toy_description` — описание внешности игрушки по фото родителя: тот же
    источник правды, что у экстрактора и судьи. При его наличии лист
    получает требение следовать фотографии, при отсутствии (CLI-сценарий)
    промпт собирается ровно как раньше.

    Чистая детерминированная функция: без LLM и ввода-вывода.
    """

    characters = CHARACTER_SHEET_CHARACTER_SEPARATOR.join(
        f"{character.name}: {character.full_desc}"
        for character in entities.characters
    )
    return _join_segments(
        (
            _resolve_art_style(entities),
            CHARACTER_SHEET_HEADER,
            characters,
            _toy_reference_segment(toy_description),
            CHARACTER_SHEET_BACKGROUND,
            CHARACTER_SHEET_TAIL,
        )
    )


__all__ = [
    "CHARACTER_SHEET_BACKGROUND",
    "CHARACTER_SHEET_CHARACTER_SEPARATOR",
    "CHARACTER_SHEET_HEADER",
    "CHARACTER_SHEET_PHOTO_REFERENCE",
    "CHARACTER_SHEET_TAIL",
    "CONSISTENCY_TAIL",
    "NO_TEXT_TAIL",
    "SEGMENT_SEPARATOR",
    "build_art_prompt",
    "build_character_sheet_prompt",
]
