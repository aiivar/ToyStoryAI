"""Тесты измерений манеры (`app/story_pipeline/style_metrics.py`)."""

from __future__ import annotations

import pytest

from app.story_pipeline.models import PageText
from app.story_pipeline.style import JUNIOR
from app.story_pipeline.style_metrics import (
    MIN_SAMPLE_WORDS,
    count_diminutives,
    count_ellipsis,
    is_dialogue_sentence,
    measure_style,
    split_sentences,
    words,
)


def _pages(*texts: str) -> list[PageText]:
    """Страницы с заданным текстом и последовательной нумерацией."""

    return [
        PageText(page_num=number, text=text)
        for number, text in enumerate(texts, start=1)
    ]


def _uniform_text(words_per_sentence: int, sentences: int) -> str:
    """Текст с одинаковой длиной каждой фразы.

    Нужен для проверки порога: средняя длина фразы известна точно, поэтому
    тест не зависит от округлений живого текста.
    """

    sentence = " ".join(["а"] * words_per_sentence) + "."
    return " ".join([sentence] * sentences)


def test_split_sentences_handles_all_end_marks_and_empty_text() -> None:
    """Деление идёт по точке, восклицанию, вопросу и многоточию."""

    text = "Раз. Два! Три? Четыре… Пять"

    assert split_sentences(text) == ["Раз.", "Два!", "Три?", "Четыре…", "Пять"]
    assert split_sentences("   \n  ") == []


def test_words_ignores_punctuation_and_normalizes_case() -> None:
    """Слова считаются без знаков препинания и в нижнем регистре."""

    assert words("Мишка, ЗАЯЦ и ёжик!") == ["мишка", "заяц", "и", "ёжик"]
    assert words("") == []


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("— Здравствуй, мишка!", True),
        ("– Куда ты идёшь?", True),
        ('"Я тут", — сказал он.', True),
        ("«Кто там?» — спросил заяц.", True),
        ("Мишка вышел из шкафа.", False),
    ],
)
def test_dialogue_sentence_recognizes_dash_and_quotes(
    sentence: str, expected: bool
) -> None:
    """Реплика узнаётся по тире или кавычке в начале предложения."""

    assert is_dialogue_sentence(sentence) is expected


def test_diminutives_count_true_forms_and_skip_stoplist() -> None:
    """Уменьшительные считаются, а обычные слова вроде «ручка» — нет."""

    tokens = words(
        "Зайчик, цветочек, мишенька, ручка, ножка, девочка, бабушка, урок."
    )

    assert count_diminutives(tokens) == 3


def test_diminutives_skip_short_tokens() -> None:
    """Слишком короткие слова не дают ложных срабатываний по суффиксу."""

    assert count_diminutives(["ик", "ёк"]) == 0


def test_ellipsis_counts_both_spellings() -> None:
    """Многоточие считается и символом, и тремя точками."""

    assert count_ellipsis("Раз… два... три…") == 3


def test_empty_text_has_no_measurements_and_no_sample() -> None:
    """Пустой текст: нули, выборки нет, отклонений нет."""

    report = measure_style([], JUNIOR, "first_person")

    assert report.words == 0
    assert report.sufficient_sample is False
    assert report.deviations == []
    assert set(report.measured) == {
        "sentence_words",
        "dialogue_share",
        "diminutives_per_1000",
        "modality_per_1000",
        "ellipsis_per_1000",
        "short_sentence_share",
        "attribution_per_1000",
    }
    assert all(value == 0.0 for value in report.measured.values())


def test_small_sample_is_not_judged() -> None:
    """Текст короче порога не получает отклонений даже при явных перекосах."""

    text = _uniform_text(30, 2)

    report = measure_style(_pages(text), JUNIOR, "first_person")

    assert report.words < MIN_SAMPLE_WORDS
    assert report.sufficient_sample is False
    assert report.deviations == []
    assert report.measured["sentence_words"] == 30.0


def test_measurements_cover_all_pages_not_one() -> None:
    """Замеры считаются по объединённому тексту всех страниц."""

    first = _uniform_text(7, 20)
    second = _uniform_text(7, 20)

    report = measure_style(_pages(first, second), JUNIOR, "third_person_limited")

    assert report.words == 280
    assert report.sufficient_sample is True
    assert report.narrator == "third_person_limited"
    assert report.profile_id == "junior"


def test_value_exactly_on_gross_boundary_is_not_a_deviation() -> None:
    """Значение ровно на границе запаса — ещё не грубое расхождение."""

    # Верхняя граница длины фразы 10 слов, запас 20 % → порог 12.
    report = measure_style(
        _pages(_uniform_text(12, 21)), JUNIOR, "first_person"
    )

    assert report.words == 252
    assert report.measured["sentence_words"] == 12.0
    assert not any("длина фразы" in item for item in report.deviations)


def test_value_above_gross_boundary_is_a_deviation() -> None:
    """Средняя фраза длиннее порога даёт отклонение с фактическим значением."""

    report = measure_style(
        _pages(_uniform_text(13, 20)), JUNIOR, "first_person"
    )

    assert report.measured["sentence_words"] == 13.0
    assert any("длина фразы" in item for item in report.deviations)


def test_missing_dialogue_is_a_deviation_on_large_sample() -> None:
    """Отсутствие реплик на достаточной выборке видно как отклонение."""

    report = measure_style(
        _pages(_uniform_text(8, 32)), JUNIOR, "first_person"
    )

    assert report.measured["dialogue_share"] == 0.0
    assert any("доля реплик" in item for item in report.deviations)


def test_dialogue_modality_and_attribution_are_counted_together() -> None:
    """Реплика, модальная оговорка и атрибуция считаются в одном тексте."""

    report = measure_style(
        _pages("— Наверное, — сказал мишка. " * 40), JUNIOR, "first_person"
    )

    assert report.measured["dialogue_share"] == 1.0
    assert report.measured["modality_per_1000"] > 0
    assert report.measured["attribution_per_1000"] > 0


def test_report_is_deterministic() -> None:
    """Повторный замер того же текста даёт тот же отчёт."""

    pages = _pages(_uniform_text(8, 40))

    assert measure_style(pages, JUNIOR, "first_person") == measure_style(
        pages, JUNIOR, "first_person"
    )


def test_deviations_are_human_readable_strings() -> None:
    """Отклонения — готовые строки для артефакта и починочного запроса."""

    report = measure_style(
        _pages(_uniform_text(30, 20)), JUNIOR, "first_person"
    )

    assert report.deviations
    assert all(isinstance(item, str) and item for item in report.deviations)
    assert all("—" in item for item in report.deviations)
