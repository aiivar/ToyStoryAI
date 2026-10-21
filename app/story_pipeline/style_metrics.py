"""Измерения манеры по готовому тексту сказки.

Модуль отвечает на вопрос «насколько текст похож на манеру профиля» числами,
а не вкусом модели. Это сигнал, а не гейт: отклонения попадают в артефакт
прогона, в лог и в один починочный запрос сценариста, но не меняют статус
прогона и не могут лишить родителя книжки.

Замеры считаются по всей сказке. На странице в 60–120 слов доля реплик и
диминутивы скачут так, что любое отклонение было бы случайным шумом.

Эвристики (реплика по тире, диминутивы по суффиксам) намеренно простые:
точность здесь не нужна, нужна воспроизводимость. Поэтому у счётчиков есть
стоп-листы обычных слов, а порог «грубости» задан с запасом.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.story_pipeline.models import Narrator, PageText, StyleReport
from app.story_pipeline.prompts import (
    STYLE_DEVIATION_ABOVE,
    STYLE_DEVIATION_BELOW,
    STYLE_METRIC_LABELS,
)
from app.story_pipeline.style import StyleProfile

# Порог достаточной выборки. Меньше — замеры неизмеримы, и отклонения пусты:
# иначе короткий текст систематически давал бы ложные «нарушения».
MIN_SAMPLE_WORDS = 250

# Запас «грубости» по границам ориентира: 20 % наружу. Меньшие расхождения —
# живой разброс, а не повод тратить платный вызов.
GROSS_MARGIN = 0.2

# Атрибуция — не диапазон, а потолок: её превышение вдвое считается грубым,
# потому что «сказал» на каждой реплике убивает диалог как двигатель сцены.
ATTRIBUTION_GROSS_FACTOR = 2.0

# Слово — кириллическая последовательность, возможно с дефисом.
_WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.IGNORECASE)

# Конец предложения: точка, восклицательный, вопросительный знак, многоточие.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# Начало реплики: тире (все варианты), кавычка-ёлочка или обычная кавычка.
_DIALOGUE_STARTS = ("—", "–", "-", "«", "“", '"', "„")

# Модальные оговорки: ребёнок не утверждает, а предполагает и проверяет.
_MODALITY_MARKERS = frozenset(
    {"наверное", "наверно", "кажется", "пожалуй", "просто", "конечно", "вроде"}
)

# Глаголы атрибуции речи. Держим формы, которые реально встречаются в сказке:
# счётчик нужен для доли, а не для лемматизации.
_ATTRIBUTION_VERBS = frozenset(
    {
        "сказал",
        "сказала",
        "сказали",
        "спросил",
        "спросила",
        "спросили",
        "ответил",
        "ответила",
        "ответили",
        "прошептал",
        "прошептала",
        "крикнул",
        "крикнула",
        "подумал",
        "подумала",
        "добавил",
        "добавила",
        "повторил",
        "повторила",
    }
)

# Суффиксы уменьшительных форм. Список неполный по определению: задача —
# увидеть порядок величины, а не разобрать морфологию.
_DIMINUTIVE_SUFFIXES = (
    "ечк",
    "очк",
    "еньк",
    "оньк",
    "ушк",
    "юшк",
    "ышк",
    "ишк",
    "чик",
    "ик",
    "ёк",
    "ек",
)

# Обычные слова, которые формально похожи на уменьшительные, но не создают
# «сюсюканья»: ручка и ножка — нормативный бытовой словарь, а не приём.
_DIMINUTIVE_STOPLIST = frozenset(
    {
        "ручка",
        "ножка",
        "девочка",
        "мальчик",
        "бабушка",
        "блик",
        "пик",
        "язык",
        "урок",
        "щенок",
        "поток",
        "молоток",
        "подарок",
        "срок",
        "кусок",
        "носок",
        "песок",
        "ребёнок",
        "ребенок",
        "человек",
        "звонок",
    }
)


def split_sentences(text: str) -> list[str]:
    """Разбить текст на предложения по знакам конца фразы.

    Пустые фрагменты отбрасываются: текст страниц приходит без гарантий
    форматирования, и двойной пробел не должен считаться предложением.
    """

    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return []
    parts = _SENTENCE_SPLIT_RE.split(normalized)
    return [part.strip() for part in parts if part.strip()]


def words(text: str) -> list[str]:
    """Слова текста в нижнем регистре — тот же счёт, что у объёма страницы."""

    return _WORD_RE.findall(text.lower())


def is_dialogue_sentence(sentence: str) -> bool:
    """Начинается ли предложение как реплика персонажа."""

    stripped = sentence.strip()
    return stripped.startswith(_DIALOGUE_STARTS)


def count_diminutives(tokens: Sequence[str]) -> int:
    """Сколько слов похожи на уменьшительные формы.

    Считаем суффиксным совпадением со стоп-листом обычных слов: ошибки в обе
    стороны неизбежны, поэтому счётчик живёт как ориентир, а не как вердикт.
    Женские и средние формы проверяются без конечной гласной: «мишенька» —
    это «мишеньк» + «а», иначе половина уменьшительных терялась бы.
    """

    total = 0
    for token in tokens:
        if len(token) < 4 or token in _DIMINUTIVE_STOPLIST:
            continue
        if token.endswith(_DIMINUTIVE_SUFFIXES) or token[:-1].endswith(
            _DIMINUTIVE_SUFFIXES
        ):
            total += 1
    return total


def count_ellipsis(text: str) -> int:
    """Сколько раз в тексте встречается многоточие."""

    return text.count("…") + text.count("...")


def measure_style(
    pages: Sequence[PageText],
    profile: StyleProfile,
    narrator: Narrator,
) -> StyleReport:
    """Собрать замеры манеры по всем страницам и найти грубые отклонения."""

    text = " ".join(page.text for page in pages)
    sentence_list = split_sentences(text)
    token_list = words(text)
    word_count = len(token_list)
    sentence_count = len(sentence_list)

    per_1000 = (word_count / 1000) if word_count else 0.0
    measured = {
        "sentence_words": (
            round(word_count / sentence_count, 2) if sentence_count else 0.0
        ),
        "dialogue_share": (
            round(
                sum(
                    1
                    for sentence in sentence_list
                    if is_dialogue_sentence(sentence)
                )
                / sentence_count,
                4,
            )
            if sentence_count
            else 0.0
        ),
        "diminutives_per_1000": (
            round(count_diminutives(token_list) / per_1000, 2)
            if per_1000
            else 0.0
        ),
        "modality_per_1000": (
            round(
                sum(
                    1 for token in token_list if token in _MODALITY_MARKERS
                )
                / per_1000,
                2,
            )
            if per_1000
            else 0.0
        ),
        "ellipsis_per_1000": (
            round(count_ellipsis(text) / per_1000, 2) if per_1000 else 0.0
        ),
        "short_sentence_share": (
            round(
                sum(
                    1
                    for sentence in sentence_list
                    if len(words(sentence)) <= 2
                )
                / sentence_count,
                4,
            )
            if sentence_count
            else 0.0
        ),
        "attribution_per_1000": (
            round(
                sum(1 for token in token_list if token in _ATTRIBUTION_VERBS)
                / per_1000,
                2,
            )
            if per_1000
            else 0.0
        ),
    }

    sufficient = word_count >= MIN_SAMPLE_WORDS
    return StyleReport(
        profile_id=profile.id,
        narrator=narrator,
        words=word_count,
        measured=measured,
        deviations=_deviations(measured, profile) if sufficient else [],
        sufficient_sample=sufficient,
    )


def _deviations(
    measured: dict[str, float],
    profile: StyleProfile,
) -> list[str]:
    """Грубые расхождения замеров с ориентирами профиля.

    Снизу проверяется «ниже ориентира с запасом», сверху — «выше ориентира с
    запасом»; для атрибуции сверху — превышение потолка вдвое. Формулировки
    берутся из `prompts.py`, чтобы текст отклонения не расходился между
    логом, артефактом и починочным запросом.
    """

    norms = profile.norms
    checks = (
        ("sentence_words", norms.sentence_words, "both"),
        ("dialogue_share", norms.dialogue_share, "both"),
        ("diminutives_per_1000", norms.diminutives_per_1000, "both"),
        ("modality_per_1000", norms.modality_per_1000, "both"),
        ("ellipsis_per_1000", norms.ellipsis_per_1000, "both"),
        ("short_sentence_share", norms.short_sentence_share, "both"),
    )

    found: list[str] = []
    for key, (low, high), direction in checks:
        value = measured.get(key, 0.0)
        label = STYLE_METRIC_LABELS[key]
        if direction in ("both", "below") and value < low * (1 - GROSS_MARGIN):
            found.append(
                STYLE_DEVIATION_BELOW.format(
                    label=label, value=value, low=round(low, 2)
                )
            )
        if direction in ("both", "above") and value > high * (1 + GROSS_MARGIN):
            found.append(
                STYLE_DEVIATION_ABOVE.format(
                    label=label, value=value, high=round(high, 2)
                )
            )

    attribution = measured.get("attribution_per_1000", 0.0)
    if attribution > norms.attribution_per_1000_max * ATTRIBUTION_GROSS_FACTOR:
        found.append(
            STYLE_DEVIATION_ABOVE.format(
                label=STYLE_METRIC_LABELS["attribution_per_1000"],
                value=attribution,
                high=norms.attribution_per_1000_max,
            )
        )
    return found


__all__ = [
    "MIN_SAMPLE_WORDS",
    "count_diminutives",
    "count_ellipsis",
    "is_dialogue_sentence",
    "measure_style",
    "split_sentences",
    "words",
]
