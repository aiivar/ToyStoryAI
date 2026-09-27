"""Оркестратор пайплайна сказок: явный граф этапов и наблюдаемость прогона.

`StoryPipeline` собирает граф ``Plan → Script → Extract → Scene(repair) →
QualityReviewLoop → ImageGenerator`` из инъектированных зависимостей и
прогоняет его над единым `StoryState`. Оркестратор не создаёт HTTP-клиентов
и не читает секреты: `LLMClient` и `ImageProvider` приходят снаружи
(задача 9.3). Узел «сборка промптов/лист персонажей» реализован внутри
`ImageGeneratorStage` (M7), поэтому отдельных узлов для него нет.

Каждый этап получает текущее состояние и обогащает его; вход/выход, время и
токены пишутся в лог через `log_stage`, а метрики накапливаются самими
этапами в `state.metrics`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from loguru import logger

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import Stage, StoryState
from app.story_pipeline.observability import log_stage, total_usage
from app.story_pipeline.providers.image import ImageProvider
from app.story_pipeline.providers.llm import LLMClient
from app.story_pipeline.settings import StorySettings
from app.story_pipeline.stages.entity_extractor import EntityExtractorStage
from app.story_pipeline.stages.image_generator import ImageGeneratorStage
from app.story_pipeline.stages.judge import JudgeStage
from app.story_pipeline.stages.planner import PlannerStage
from app.story_pipeline.stages.quality_loop import QualityReviewLoop
from app.story_pipeline.stages.reviser import ReviserStage
from app.story_pipeline.stages.scene_breakdown import SceneBreakdownStage
from app.story_pipeline.stages.screenwriter import ScreenwriterStage


class StoryPipeline:
    """Явный граф этапов сказки с инъектированными провайдерами."""

    def __init__(
        self,
        llm: LLMClient,
        image_provider: ImageProvider,
        settings: StorySettings,
        *,
        owns_providers: bool = False,
    ) -> None:
        self._llm = llm
        self._image_provider = image_provider
        self._settings = settings
        self._owns_providers = owns_providers
        self._stages: list[Stage] = [
            PlannerStage(llm),
            ScreenwriterStage(llm),
            EntityExtractorStage(llm),
            SceneBreakdownStage(llm),
            QualityReviewLoop(
                JudgeStage(llm),
                ReviserStage(llm),
                max_revisions=settings.story_max_revisions,
            ),
            ImageGeneratorStage(image_provider),
        ]

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Узлы графа в порядке выполнения (для наблюдаемости и тестов)."""

        return tuple(self._stages)

    async def run(self, state: StoryState) -> StoryState:
        """Прогнать весь граф над состоянием и вернуть его же.

        Метки `started_at`/`finished_at` проставляет оркестратор: он — точка
        начала и конца прогона, поэтому время осмысленно и при сбое.

        Единственное место установки `status="failed"` — здесь. Этапы
        деградации (`needs_review` при исчерпании ревизий или сбое страниц)
        помечают результат сами и не бросают исключений; `failed` же означает,
        что прогон не смог выдать результат и завершился типизированной
        ошибкой, которую оркестратор пробрасывает наружу без подмены типа.
        """

        state.started_at = datetime.now(timezone.utc)
        state.output_dir = self._settings.story_output_dir
        current_stage = "?"
        try:
            for stage in self._stages:
                current_stage = stage.name
                await self._run_stage(stage, state)
        except StoryPipelineError as exc:
            state.status = "failed"
            logger.error(
                "pipeline failed at stage {}: {}", current_stage, exc
            )
            raise
        finally:
            state.finished_at = datetime.now(timezone.utc)
        return state

    async def aclose(self) -> None:
        """Закрыть провайдеры, которыми владеет оркестратор.

        Провайдеры инъектируются снаружи и по умолчанию остаются
        собственностью вызывающего (CLI закрывает их сам), поэтому
        `owns_providers=False` (дефолт) делает метод безопасным no-op: чужие
        HTTP-клиенты не вырываются из-под ног. При `owns_providers=True`
        закрываются только провайдеры, поддерживающие `aclose` (mock
        пропускается), и сбой закрытия одного не мешает закрыть второй:
        ошибки собираются, первая всплывает после попытки закрыть все.
        """

        if not self._owns_providers:
            return
        first_error: Exception | None = None
        for provider in (self._llm, self._image_provider):
            close = getattr(provider, "aclose", None)
            if not callable(close):
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - закрыть и остальные
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def _run_stage(self, stage: Stage, state: StoryState) -> None:
        """Выполнить узел, залогировав вход/выход, время и расход токенов."""

        summary_in = _fingerprint(state)
        metrics_before = len(state.metrics)
        started = time.perf_counter()
        await stage.run(state)
        duration_ms = (time.perf_counter() - started) * 1000
        log_stage(
            stage.name,
            summary_in,
            _fingerprint(state),
            duration_ms,
            total_usage(state.metrics[metrics_before:]),
        )


def _fingerprint(state: StoryState) -> str:
    """Краткая сводка состояния для лога: без промптов и секретов."""

    return (
        f"brief={_flag(state.brief)} pages={len(state.pages)} "
        f"entities={_flag(state.entities)} scenes={len(state.scenes)} "
        f"images={len(state.images)} reviews={state.revision_count} "
        f"status={state.status}"
    )


def _flag(value: object) -> str:
    """Отрендерить наличие артефакта как yes/no."""

    return "yes" if value is not None else "no"


__all__ = ["StoryPipeline"]
