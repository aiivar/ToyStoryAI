"""README описывает текущий флоу «сказка-книжка» (задача 8.2).

Документация протухает тихо: код меняется, а README остаётся правдоподобным.
Поэтому факты, на которые читатель пойдёт смотреть в репозиторий или в
платёжный кабинет, зафиксированы тестом: имена этапов и `on_stage` (на них
висят тексты прогресса), переменные `ROUTERAI_*`, ориентир по стоимости
одной сказки и ручной порядок проверки на реальном ключе.
"""

from __future__ import annotations

from pathlib import Path

from app.story_pipeline.settings import StorySettings
from app.story_pipeline.stages.entity_extractor import EntityExtractorStage
from app.story_pipeline.stages.image_generator import ImageGeneratorStage
from app.story_pipeline.stages.judge import JudgeStage
from app.story_pipeline.stages.planner import PlannerStage
from app.story_pipeline.stages.quality_loop import QualityReviewLoop
from app.story_pipeline.stages.reviser import ReviserStage
from app.story_pipeline.stages.scene_breakdown import SceneBreakdownStage
from app.story_pipeline.stages.screenwriter import ScreenwriterStage

README = Path(__file__).resolve().parent.parent / "README.md"

# Узлы графа в порядке работы пайплайна: петля качества внутри себя
# разворачивает `judge` и `reviser` попарно.
STAGE_NAMES = (
    PlannerStage.name,
    ScreenwriterStage.name,
    EntityExtractorStage.name,
    SceneBreakdownStage.name,
    QualityReviewLoop.name,
    ImageGeneratorStage.name,
)

# Стоимость одной сказки: лист персонажей плюс столько картинок, сколько
# разрешает бюджет иллюстраций. Раньше бюджета не было, и число картинок
# равнялось числу страниц; теперь они независимы, поэтому берём бюджет.
IMAGES_PER_STORY = StorySettings.model_fields["story_illustrated_pages"].default + 1


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_lists_every_pipeline_stage() -> None:
    """README перечисляет имена этапов — по ним идёт `on_stage`."""

    text = _readme()

    missing = [name for name in STAGE_NAMES if name not in text]

    assert missing == [], f"README не упоминает этапы: {missing}"


def test_readme_documents_stage_callback_and_progress_texts() -> None:
    """README объясняет `on_stage` и тексты, которые он показывает."""

    text = _readme()

    assert "on_stage" in text
    assert "Готовлю сказку" in text
    assert "Рисую страницы" in text


def test_readme_documents_stage_name_reviser_and_judge() -> None:
    """Внутренние имена этапов петли качества тоже документированы."""

    text = _readme()

    assert JudgeStage.name in text
    assert ReviserStage.name in text


def test_readme_uses_routerai_settings_names() -> None:
    """Настройки иллюстраций документированы под `ROUTERAI_*`, не `KLING_*`."""

    text = _readme()

    assert "ROUTERAI_API_KEY" in text
    assert "ROUTERAI_IMAGE_MODEL" in text
    assert "KLING_" not in text


def test_readme_states_image_cost_per_story() -> None:
    """В README есть ориентир: столько генераций изображений стоит сказка."""

    text = _readme()

    assert str(IMAGES_PER_STORY) in text
    assert "генерац" in text


def test_readme_documents_manual_check_order_on_real_key() -> None:
    """Описан ручной порядок проверки на реальном ключе."""

    text = _readme()

    assert "character_sheet" in text
    assert "page_" in text
    assert "story.json" in text


def test_readme_describes_the_whole_bot_dialogue() -> None:
    """README описывает флоу бота целиком: фото → возраст → мораль → выдача.

    Пользовательский маршрут — первое, что ищут в README. Перечисление шагов
    проверяется по порядку, чтобы перестановка абзацев не прошла молча.
    """

    text = _readme()

    steps = ("фото", "возраст", "мораль", "страниц")
    positions = []
    for step in steps:
        index = text.lower().find(step)
        assert index != -1, f"README не описывает шаг «{step}»"
        positions.append(index)

    assert positions == sorted(positions), (
        "шаги диалога описаны не в порядке пользовательского маршрута"
    )


def test_readme_points_at_real_key_setup() -> None:
    """README говорит, какие ключи нужны для живой проверки."""

    text = _readme()

    assert "DEEPSEEK_API_KEY" in text
    assert "ROUTERAI_API_KEY" in text
    assert "python -m app.main" in text
