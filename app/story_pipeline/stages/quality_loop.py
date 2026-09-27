"""Петля контроля качества: судья → ревизор → судья с жёстким бюджетом.

Этап-подобный компонент: у него есть `name` и `run(state)`, поэтому
оркестратор встраивает его в граф как обычный узел. Судья и ревизор
внедряются в конструктор, чтобы петля не знала деталей их реализации и
легко подменялась фейками в тестах.

Бюджет ревизий ограничен: при исчерпании лимита прогон не падает, а
сохраняется со статусом `needs_review`. Все вердикты копятся в
`state.judge_history` — по ним считается доля прохождения без ревизии.
"""

from __future__ import annotations

from app.story_pipeline.models import StoryState
from app.story_pipeline.stages.judge import JudgeStage, validate_judge_verdict
from app.story_pipeline.stages.reviser import ReviserStage

# Лимит циклов «судья → ревизор → судья» из контракта пайплайна
# (`StorySettings.story_max_revisions`). Оркестратор может переопределить.
DEFAULT_MAX_REVISIONS = 2


class QualityReviewLoop:
    """Узел графа «Контроль качества»: гоняет судью и ревизора до вердикта."""

    name = "quality"

    def __init__(
        self,
        judge: JudgeStage,
        reviser: ReviserStage,
        *,
        max_revisions: int = DEFAULT_MAX_REVISIONS,
    ) -> None:
        self._judge = judge
        self._reviser = reviser
        self._max_revisions = max_revisions

    async def run(self, state: StoryState) -> StoryState:
        """Проверять сказку, пока судья не пропустит её или не кончится бюджет.

        Страницы из `state.unresolved_pages` (остаток неизвестных имён
        после раскадровки) считаются заведомо проблемными: судья может их
        не увидеть, но петля обязана их отревизовать.
        """

        while True:
            await self._judge.run(state)
            verdict = validate_judge_verdict(
                state.judge_history[-1], state.pages
            )
            if verdict.passed and not state.unresolved_pages:
                return state
            if state.revision_count >= self._max_revisions:
                # Бюджет исчерпан, а проверка не пройдена: сохраняем что
                # есть и помечаем прогон для ручного разбора.
                state.status = "needs_review"
                return state
            await self._reviser.run(state)
            state.revision_count += 1


__all__ = ["DEFAULT_MAX_REVISIONS", "QualityReviewLoop"]
