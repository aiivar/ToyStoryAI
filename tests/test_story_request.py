"""Тесты сборки входа пайплайна: тема, описание игрушки, каталог прогона.

Задачи 5.3 и 5.9: тема и описание собираются детерминированно из
распознанного описания игрушки, выбранной морали и возраста — свободного
текста от модели в них нет, поэтому одинаковые входные данные всегда дают
одинаковый вход прогона.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.bot.story_request import (
    build_run_dir,
    build_theme,
    build_toy_description,
    join_pages_text,
)
from app.services.toy_vision import ToyDescription
from app.story_pipeline.settings import StorySettings

DESCRIPTION = ToyDescription(
    kind="плюшевый медвежонок",
    color="рыжий с белым пятном на груди",
    material="мягкая ткань",
    features="кисточка на голове, красный шарф, круглые лапы",
    short="рыжий плюшевый медвежонок в красном шарфе",
)

MORAL = "делиться с другими"


def test_toy_description_is_deterministic_assembly_of_fields() -> None:
    """Описание игрушки — сборка полей `ToyDescription` без домыслов."""

    assert build_toy_description(DESCRIPTION) == (
        "Вид: плюшевый медвежонок. Цвет: рыжий с белым пятном на груди. "
        "Материал: мягкая ткань. Особенности: кисточка на голове, "
        "красный шарф, круглые лапы."
    )


def test_theme_contains_toy_short_moral_and_age() -> None:
    """Тема сказки включает описание игрушки, мораль и возраст ребёнка."""

    theme = build_theme(DESCRIPTION, MORAL, 5)

    assert theme == (
        "Сказка о любимой игрушке: рыжий плюшевый медвежонок в красном "
        "шарфе. "
        "Главный герой — эта игрушка, и её облик не меняется: "
        "Вид: плюшевый медвежонок. Цвет: рыжий с белым пятном на груди. "
        "Материал: мягкая ткань. Особенности: кисточка на голове, "
        "красный шарф, круглые лапы. "
        f"Мораль сказки: {MORAL}. "
        "Читатель: ребёнок 5 лет, простые короткие предложения."
    )


def test_theme_is_stable_for_same_input() -> None:
    """Повторная сборка темы даёт ту же строку — детерминизм, а не словарь."""

    assert build_theme(DESCRIPTION, MORAL, 5) == build_theme(DESCRIPTION, MORAL, 5)


def test_run_dir_is_unique_per_user_and_time() -> None:
    """Каталог прогона включает пользователя и метку времени."""

    settings = StorySettings(_env_file=None)  # type: ignore[call-arg]
    first = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    second = datetime(2026, 3, 4, 5, 6, 8, tzinfo=timezone.utc)

    run_dir = build_run_dir(7, settings, first)
    another = build_run_dir(7, settings, second)
    other_user = build_run_dir(8, settings, first)

    assert run_dir.startswith(str(Path("output") / "7-"))
    assert run_dir != another
    assert other_user != run_dir
    assert run_dir.startswith(str(settings.story_output_dir))


def test_join_pages_text_glues_pages_into_one_paragraph() -> None:
    """Полный текст сказки — абзац из текстов страниц по порядку номеров."""

    pages = [
        _page(2, "Второе."),
        _page(1, "Первое."),
    ]

    assert join_pages_text(pages) == "Первое. Второе."


def _page(page_num: int, text: str):
    from app.story_pipeline.models import PageText

    return PageText(page_num=page_num, text=text)
