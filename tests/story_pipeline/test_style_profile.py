"""Тесты профилей манеры (`app/story_pipeline/style.py`)."""

from __future__ import annotations

import pytest

from app.story_pipeline.errors import PipelineValidationError
from app.story_pipeline.style import (
    JUNIOR,
    PROFILES,
    SENIOR,
    STYLE_AGE_MAX,
    STYLE_AGE_MIN,
    resolve_style_profile,
)

# Обороты имитации, которых в профиле быть не должно: он описывает механику
# приёмов, а не личность или источник подражания.
_FORBIDDEN_PHRASES = ("по мотивам", "в стиле ", "как у ")


@pytest.mark.parametrize("age", [3, 4, 5, 6])
def test_junior_profile_covers_preschool_ages(age: int) -> None:
    """Возраст 3–6 получает младший профиль."""

    assert resolve_style_profile(age) is JUNIOR


@pytest.mark.parametrize("age", [7, 8, 10, 12])
def test_senior_profile_covers_school_ages(age: int) -> None:
    """Возраст 7–12 получает старший профиль."""

    assert resolve_style_profile(age) is SENIOR


@pytest.mark.parametrize("age", [STYLE_AGE_MIN - 1, STYLE_AGE_MAX + 1, 0, 99])
def test_age_outside_supported_range_is_rejected(age: int) -> None:
    """Возраст вне 3–12 — ошибка входа, а не молчаливый младший профиль."""

    with pytest.raises(PipelineValidationError) as failure:
        resolve_style_profile(age)

    message = str(failure.value)
    assert str(age) in message
    assert f"{STYLE_AGE_MIN}..{STYLE_AGE_MAX}" in message


def test_profiles_cover_range_without_gaps() -> None:
    """Профили идут подряд и покрывают весь поддерживаемый диапазон."""

    assert PROFILES[0].age_min == STYLE_AGE_MIN
    assert PROFILES[-1].age_max == STYLE_AGE_MAX
    for previous, following in zip(PROFILES, PROFILES[1:]):
        assert previous.age_max + 1 == following.age_min


@pytest.mark.parametrize("profile", PROFILES, ids=lambda item: item.id)
def test_profile_has_techniques_prohibitions_and_both_openings(profile) -> None:
    """У профиля есть все части, нужные трём этапам и обоим рассказчикам."""

    assert profile.techniques
    assert profile.prohibitions
    assert set(profile.openings) == {"first_person", "third_person_limited"}
    assert all(text.strip() for text in profile.openings.values())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda item: item.id)
def test_norms_are_ordered_and_within_meaningful_bounds(profile) -> None:
    """Ориентиры заданы парами «минимум ≤ максимум» и лежат в разумных долях."""

    norms = profile.norms
    for low, high in (
        norms.sentence_words,
        norms.dialogue_share,
        norms.diminutives_per_1000,
        norms.modality_per_1000,
        norms.ellipsis_per_1000,
        norms.short_sentence_share,
    ):
        assert 0.0 <= low <= high
    assert 0.0 <= norms.dialogue_share[1] <= 1.0
    assert 0.0 <= norms.short_sentence_share[1] <= 1.0
    assert norms.attribution_per_1000_max > 0


def test_senior_allows_social_conflict_and_junior_does_not() -> None:
    """Старший профиль добавляет социальный конфликт, младший его не требует."""

    junior_text = " ".join(JUNIOR.techniques)
    senior_text = " ".join(SENIOR.techniques)

    assert "травля" not in junior_text
    assert "ярлык" not in junior_text
    assert "травля" in senior_text
    assert "ярлык" in senior_text
    assert set(JUNIOR.techniques).issubset(set(SENIOR.techniques))


@pytest.mark.parametrize("profile", PROFILES, ids=lambda item: item.id)
def test_profile_texts_are_mechanical_not_imitative(profile) -> None:
    """Профиль описывает механику приёмов, а не отсылает к источнику."""

    texts = [
        *profile.techniques,
        *profile.prohibitions,
        *profile.openings.values(),
    ]
    lowered = " ".join(texts).lower()

    for phrase in _FORBIDDEN_PHRASES:
        assert phrase not in lowered
