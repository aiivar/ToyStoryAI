"""Тесты этапа планирования (`app/story_pipeline/stages/planner.py`).

Этап проверяется через публичный интерфейс `async def run(state)`: тест
подставляет фейковый `LLMClient` и смотрит, что этап запросил у модели и
что записал в состояние. Внутренние методы этапа не мокаются.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.story_pipeline.models import NarrativeArc, StoryBrief, StoryState
from app.story_pipeline.stages.planner import PlannerStage


def _state() -> StoryState:
    """Собрать минимальное входное состояние для этапа планирования."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=5,
        moral="Смелость помогает найти друзей",
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def _brief() -> StoryBrief:
    """Заготовка брифа, которую «вернёт» модель."""

    return StoryBrief(
        logline="Плюшевый мишка идёт через лес и находит друга.",
        controlling_idea="Смелость открывает дорогу к дружбе.",
        character_arc="Мишка учится не бояться и помогает зайцу.",
        narrative_arc=NarrativeArc(
            exposition="Мишка живёт в шкафу детской.",
            rising_action="Мишка выходит в лес и слышит плач.",
            climax="Мишка спасает зайца от ручья.",
            resolution="Мишка и заяц возвращаются домой друзьями.",
        ),
    )


async def test_planner_stores_brief_returned_by_model(fake_llm) -> None:
    """Планировщик кладёт бриф из ответа модели в `state.brief`."""

    brief = _brief()
    llm = fake_llm([brief])
    stage = PlannerStage(llm)
    state = _state()

    result = await stage.run(state)

    assert result is state
    assert state.brief == brief
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is StoryBrief
    assert llm.calls[0]["temperature"] == 0.7


async def test_planner_prompt_carries_inputs_and_rendered_schema(
    fake_llm,
) -> None:
    """Промпт несёт theme/age/moral и схему из модели, метрика — с токенами."""

    llm = fake_llm([_brief()])
    stage = PlannerStage(llm)
    state = _state()

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert "Приключение плюшевого мишки" in user
    assert "5" in user
    assert "Смелость помогает найти друзей" in user
    # Маркеры отрендеренной JSON-схемы, а не рукописного описания.
    assert "$defs" in user
    assert "required" in user

    metric = state.metrics[-1]
    assert metric.stage == "plan"
    assert metric.prompt_tokens == 11
    assert metric.completion_tokens == 22
    assert metric.duration_ms >= 0
