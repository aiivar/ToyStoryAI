"""Профили манеры повествования для текстовых этапов сказки.

Профиль — типизированные данные, а не конфигурация окружения: наборы норм
меняются вместе с промптами и тестами, а `.env` должен оставаться местом
для инфраструктурных параметров. Имя автора-образца здесь не используется:
профиль описывает механику приёмов, а не личность, поэтому тексты пригодны
для подстановки в промпт без отсылок к конкретному писателю.

Модуль не собирает промпты: он отдаёт данные, а рендер блоков живёт в
`prompts.py`, рядом с остальными обращениями к модели.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.story_pipeline.errors import PipelineValidationError
from app.story_pipeline.models import Narrator

# Поддерживаемый диапазон возрастов. Шире текущих настроек пайплайна (3–6):
# старший профиль доступен прогону по контракту и тестам, а пользовательский
# сценарий для 7–12 включится, когда будут подняты границы возраста.
STYLE_AGE_MIN = 3
STYLE_AGE_MAX = 12


@dataclass(frozen=True)
class StyleNorms:
    """Числовые ориентиры стиля по всей сказке.

    Диапазоны, а не точные значения: текст живой, и попадание в интервал —
    признак манеры, а не выполнения норматива. Все замеры считаются на 1000
    слов, кроме длины фразы и долей, чтобы объём сказки не влиял на оценку.
    """

    sentence_words: tuple[float, float]
    dialogue_share: tuple[float, float]
    diminutives_per_1000: tuple[float, float]
    modality_per_1000: tuple[float, float]
    ellipsis_per_1000: tuple[float, float]
    short_sentence_share: tuple[float, float]
    attribution_per_1000_max: float


@dataclass(frozen=True)
class StyleProfile:
    """Возрастной набор приёмов, запретов и ориентиров.

    `techniques` и `prohibitions` — короткие требования, которые рендерятся
    в промпт планировщика, сценариста и ревизора. `openings` — формулировка
    представления героя под выбранного рассказчика: самопредставление
    игрушки и третье лицо с фокалом требуют разных слов, и собирать их
    ветвлением в этапах значило бы держать стиль в двух местах.
    """

    id: str
    age_min: int
    age_max: int
    norms: StyleNorms
    techniques: tuple[str, ...]
    prohibitions: tuple[str, ...]
    openings: Mapping[Narrator, str]


# Запреты общие для возрастов: они и есть антидидактика, без которой
# приёмы превращаются в мораль в обёртке.
_COMMON_PROHIBITIONS: tuple[str, ...] = (
    "Не проговаривай мораль: ни в финале, ни в реплике взрослого. "
    "Смысл выводится из цены поступка.",
    "Не спасай мир: масштаб поступка — детский. Бытовая ставка важнее "
    "глобальной катастрофы.",
    "Не делай взрослого всезнающим: он может ошибаться и называть вещи "
    "не своими именами.",
    "Не объясняй мир энциклопедически и не давай длинных описаний: "
    "пейзаж — одна–три фразы через восприятие героя.",
    "Диминутивы принадлежат речи персонажей, а не автору: автор не "
    "сюсюкает.",
)

_OPENINGS: Mapping[Narrator, str] = {
    "first_person": (
        "Повествование идёт от первого лица: игрушка сама себя "
        "представляет — имя, как её зовут на самом деле, кому она "
        "принадлежит, что любит."
    ),
    "third_person_limited": (
        "Повествование идёт в третьем лице с фокалом на игрушке: герой "
        "назван и сразу показан в действии, его мысли и чувства доступны, "
        "но о себе он не рассказывает."
    ),
}

_STRUCTURE_COMMON: tuple[str, ...] = (
    "Герой попадает в чужой уклад — двор, дачу, лес, чужую комнату — "
    "и осваивается в нём.",
    "Цель героя бытовая и маленькая: дойти, доплыть, отдать, признаться, "
    "промолчать. Цена решения — репутационная, а не мировая.",
    "Ключевую правду герой добывает сам: подслушивает, находит письмо или "
    "замечает то, что взрослые не собирались ему показывать.",
    "Эмоция показывается телом и средой, а не называется словом: "
    "зажмурился, сердце скакнуло, море стало полосатым.",
    "Трудное объясняется одной утешительной метафорой, без эвфемизмов "
    "вроде «уехал далеко».",
    "Финал — примирение или обретение, а не победа над злодеем.",
)

JUNIOR = StyleProfile(
    id="junior",
    age_min=3,
    age_max=6,
    norms=StyleNorms(
        sentence_words=(7.0, 10.0),
        dialogue_share=(0.12, 0.19),
        diminutives_per_1000=(8.0, 17.0),
        modality_per_1000=(3.0, 5.0),
        ellipsis_per_1000=(5.0, 10.0),
        short_sentence_share=(0.04, 0.07),
        attribution_per_1000_max=6.0,
    ),
    techniques=_STRUCTURE_COMMON,
    prohibitions=_COMMON_PROHIBITIONS,
    openings=_OPENINGS,
)

SENIOR = StyleProfile(
    id="senior",
    age_min=7,
    age_max=12,
    norms=StyleNorms(
        sentence_words=(8.0, 10.0),
        dialogue_share=(0.12, 0.19),
        diminutives_per_1000=(10.0, 20.0),
        modality_per_1000=(3.0, 5.0),
        ellipsis_per_1000=(7.0, 12.0),
        short_sentence_share=(0.05, 0.07),
        attribution_per_1000_max=5.0,
    ),
    techniques=_STRUCTURE_COMMON
    + (
        "Конфликт может быть социальным — травля, исключение, ярлык — или "
        "семейным: развод, приёмная семья, бедность.",
        "У героя может быть два имени: официальное и уличное. История "
        "проверяет, какое из них правда.",
    ),
    prohibitions=_COMMON_PROHIBITIONS,
    openings=_OPENINGS,
)

PROFILES: tuple[StyleProfile, ...] = (JUNIOR, SENIOR)


def resolve_style_profile(age: int) -> StyleProfile:
    """Выбрать профиль по возрасту читателя.

    Возраст вне поддерживаемого диапазона — ошибка входа, а не повод молча
    взять младший профиль: иначе сказка для подростка писалась бы по
    дошкольным правилам, и по артефакту это выглядело бы как замысел.
    """

    for profile in PROFILES:
        if profile.age_min <= age <= profile.age_max:
            return profile
    raise PipelineValidationError(
        f"Возраст {age} вне поддерживаемого диапазона профилей манеры "
        f"{STYLE_AGE_MIN}..{STYLE_AGE_MAX}: профиль не определён."
    )


def get_style_profile(profile_id: str | None, age: int) -> StyleProfile:
    """Профиль этапа: уже разрешённый прогоном либо выбранный по возрасту.

    Второй путь нужен этапам, которые собирают состояние сами (тесты и
    внешние композиции) и не проходят через оркестратор: без него в промпт
    уехал бы пустой профиль. Неизвестный идентификатор — ошибка, а не
    молчаливая подмена: иначе артефакт врал бы о правилах прогона.
    """

    if profile_id is None:
        return resolve_style_profile(age)
    for profile in PROFILES:
        if profile.id == profile_id:
            return profile
    known = ", ".join(profile.id for profile in PROFILES)
    raise PipelineValidationError(
        f"Неизвестный профиль манеры {profile_id!r}: известны {known}."
    )


__all__ = [
    "JUNIOR",
    "PROFILES",
    "SENIOR",
    "STYLE_AGE_MAX",
    "STYLE_AGE_MIN",
    "StyleNorms",
    "StyleProfile",
    "get_style_profile",
    "resolve_style_profile",
]
