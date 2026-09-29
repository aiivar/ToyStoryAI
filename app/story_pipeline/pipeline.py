"""Оркестратор пайплайна сказок: явный граф этапов и наблюдаемость прогона.

`StoryPipeline` собирает граф ``Plan → Script → Extract → Scene(repair) →
QualityReviewLoop → ImageGenerator`` из инъектированных зависимостей и
прогоняет его над единым `StoryState`. Оркестратор не создаёт HTTP-клиентов
и не читает секреты: `LLMClient` и `ImageProvider` приходят снаружи
(задача 9.3). Узел «сборка промптов/лист персонажей» реализован внутри
`ImageGeneratorStage` (M7), поэтому отдельных узлов для него нет.

Каждый этап получает текущее состояние и обогащает его; вход/выход, время и
токены пишутся в лог через `log_stage`, а метрики накапливаются самими
этапами в `state.metrics`. Необязательный `on_stage` сообщает имя этапа на
его границе, чтобы вызывающая сторона могла рассказать пользователю, что
происходит, и `await`-ится, чтобы сообщение ушло до старта следующего этапа.

Каталог вывода берётся из настроек только если прогон не задал его сам:
уникальный каталог на прогон (например, у бота) перетирать нельзя, иначе
параллельные прогоны писали бы файлы в одно место.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from loguru import logger

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.illustrations import (
    illustrated_page_nums,
    normalize_page_nums,
)
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

# Асинхронный колбэк прогресса: принимает имя этапа, ничего не возвращает.
# `Awaitable`, а не `None`: оркестратор ждёт отправку сообщения, иначе
# «Рисую страницы…» ушло бы уже после того, как рисование началось.
StageCallback = Callable[[str], Awaitable[None]]


class StoryPipeline:
    """Явный граф этапов сказки с инъектированными провайдерами.

    `on_stage` — необязательный асинхронный колбэк, которым оркестратор
    сообщает имя этапа на его границе. Он `await`-ится, поэтому отправитель
    сообщений успевает дописать «Готовлю сказку…» до старта следующего этапа.
    Колбэк не тащит в пайплайн знание о мессенджере и по умолчанию
    отсутствует, поэтому прогон без него ничем не отличается.
    """

    def __init__(
        self,
        llm: LLMClient,
        image_provider: ImageProvider,
        settings: StorySettings,
        *,
        owns_providers: bool = False,
        reference_photo: bytes | None = None,
        on_stage: StageCallback | None = None,
    ) -> None:
        """Собрать граф этапов над инъектированными провайдерами.

        `reference_photo` — байты фото игрушки, которые прокидываются в этап
        иллюстрирования как единственный референс character sheet. В
        `StoryState` фото не попадает: состояние сериализуется в
        `story.json` и остаётся без бинарных данных, поэтому фото — вход
        прогона, а не его артефакт. `None` — сценарий без фото (CLI).
        """

        self._llm = llm
        self._image_provider = image_provider
        self._settings = settings
        self._owns_providers = owns_providers
        self._on_stage = on_stage
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
            ImageGeneratorStage(
                image_provider,
                reference_photo=reference_photo,
                aspect_ratio=settings.story_image_aspect_ratio,
            ),
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
        Наружу уходит только `StoryPipelineError`: сбой колбэка `on_stage`
        ловится в `_notify_stage` и лишь логируется.
        """

        state.started_at = datetime.now(timezone.utc)
        # Каталог из настроек — только запасной вариант: если прогон уже
        # получил свой каталог (уникальный на бота), перетирать его нельзя,
        # иначе параллельные прогоны писали бы файлы в одно место.
        if state.output_dir is None:
            state.output_dir = self._settings.story_output_dir
        # Набор иллюстрируемых страниц материализуется здесь и больше не
        # меняется: от него зависят промпт сценариста, вызовы провайдера и
        # проверка покрытия, и расхождение между этими тремя выглядело бы
        # как «картинки не те». Явно заданное поле — точка per-run
        # переопределения, но нормализуется оно тем же правилом: опечатка
        # вызывающей стороны обязана исправляться здесь, а не ронять
        # прогон позже и непонятно где.
        self._resolve_illustrated_pages(state)
        current_stage = "?"
        try:
            for stage in self._stages:
                current_stage = stage.name
                # Граница этапа: сообщение о прогрессе уходит и завершается
                # до старта следующего этапа, иначе отправитель отстал бы.
                await self._notify_stage(stage.name)
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

    def _resolve_illustrated_pages(self, state: StoryState) -> None:
        """Заполнить набор иллюстрируемых страниц, если его ещё нет.

        Пустое поле состояния означает «набор не задан»: набор вычисляется
        из бюджета настроек и числа страниц. Явно заданный набор
        сохраняется, но всё равно нормализуется — вычисленный набор
        корректен по построению, поэтому ошибаться может только тот, кто
        задал его руками.

        Логируется исходное значение, если оно было задано и не совпало с
        нормализованным: иначе прогон молча исправил бы чужие данные, и
        разбор потом не увидел бы, что было не так.
        """

        requested = list(state.illustrated_pages)
        source = (
            "настройки"
            if not requested
            else f"вызывающая сторона {requested}"
        )
        normalized = normalize_page_nums(
            requested
            if requested
            else illustrated_page_nums(
                state.num_pages,
                self._settings.story_illustrated_pages,
            ),
            state.num_pages,
        )
        if requested and list(normalized) != requested:
            logger.warning(
                "pipeline: набор иллюстрируемых страниц нормализован "
                "{} -> {}",
                requested,
                list(normalized),
            )
        state.illustrated_pages = list(normalized)
        logger.debug(
            "pipeline: иллюстрируемые страницы из {}: {}",
            source,
            list(normalized),
        )

    async def _notify_stage(self, stage_name: str) -> None:
        """Сообщить о начале этапа, если колбэк передан, и не дать сбою
        отправителя выбросить прогон.

        Вызов `await`-ится намеренно: сообщение пользователю должно уйти
        до того, как начнётся платная работа этапа. Без колбэка метод —
        дешёвый no-op, лишних await в прогоне не появляется.

        Ошибка колбэка ловится целиком и превращается в предупреждение в
        лог. Отправитель — внешний код (сеть мессенджера, таймаут), и его
        сбой ничего не говорит о качестве сказки: к моменту рисования
        деньги уже потрачены, и терять все картинки из-за неотправленного
        сообщения о прогрессе нельзя. Статус прогона при этом не
        подменяется — его выставляют этапы, поэтому ложный `ok` на
        частично неудачном прогоне появиться не может.
        """

        if self._on_stage is None:
            return
        try:
            await self._on_stage(stage_name)
        except Exception as exc:  # noqa: BLE001 - прогресс не роняет прогон
            logger.warning(
                "pipeline: on_stage callback failed before stage {}: {}",
                stage_name,
                exc,
            )

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


__all__ = ["StageCallback", "StoryPipeline"]
