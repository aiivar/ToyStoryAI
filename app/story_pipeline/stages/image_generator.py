"""Этап иллюстрирования: character sheet и PNG-страницы через провайдер.

Тонкий узел графа: собирает промпты чистыми функциями `prompt_builder`,
вызывает инъецированный `ImageProvider` и сохраняет результат в
`state.output_dir`. Сначала генерируется один reference-лист персонажей,
затем каждая страница — с ним как reference. Ретраи живут внутри
провайдера; этап отвечает за порядок, файлы и мягкую деградацию.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from app.story_pipeline.errors import (
    PermanentImageGenerationError,
    TransientImageGenerationError,
)
from app.story_pipeline.models import (
    ExtractedEntities,
    PageArt,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.providers.image import ImageProvider
from app.story_pipeline.stages.prompt_builder import (
    build_art_prompt,
    build_character_sheet_prompt,
)

# Имя файла reference-листа персонажей в каталоге вывода.
CHARACTER_SHEET_FILENAME = "character_sheet.png"


def page_filename(page_num: int) -> str:
    """Имя PNG-файла страницы по её номеру."""

    return f"page_{page_num}.png"


class ImageGeneratorStage:
    """Узел графа «Иллюстрирование»: sheet + страницы через `ImageProvider`."""

    name = "illustrate"

    def __init__(self, provider: ImageProvider) -> None:
        self._provider = provider

    async def run(self, state: StoryState) -> StoryState:
        """Сгенерировать reference-лист и страницы, записав пути и seed."""

        started = time.perf_counter()
        output_dir = self._resolve_output_dir(state)
        entities = self._require_entities(state)
        output_dir.mkdir(parents=True, exist_ok=True)

        sheet = await self._provider.generate(
            build_character_sheet_prompt(entities)
        )
        sheet_path = output_dir / CHARACTER_SHEET_FILENAME
        sheet_path.write_bytes(sheet.data)
        state.character_sheet_path = str(sheet_path)
        state.character_sheet_seed = sheet.seed

        art_prompts: list[str] = []
        failed_pages: list[int] = []
        for scene in state.scenes:
            prompt = build_art_prompt(scene, entities)
            art_prompts.append(prompt)
            try:
                image = await self._provider.generate(
                    prompt, reference_images=[sheet.data]
                )
            except TransientImageGenerationError as exc:
                # Провайдер уже исчерпал свои повторы: фиксируем пропуск
                # и продолжаем, чтобы одна страница не обнулила прогон.
                logger.warning(
                    "illustration: page {} failed after retries: {}",
                    scene.page_num,
                    exc,
                )
                failed_pages.append(scene.page_num)
                continue
            path = output_dir / page_filename(scene.page_num)
            path.write_bytes(image.data)
            state.images.append(
                PageArt(
                    page_num=scene.page_num,
                    path=str(path),
                    seed=image.seed,
                )
            )

        error = None
        if failed_pages:
            state.status = "needs_review"
            error = "не сгенерированы страницы: " + ", ".join(
                str(page_num) for page_num in failed_pages
            )
        state.art_prompts = art_prompts
        state.metrics.append(
            StageMetrics(
                stage=self.name,
                duration_ms=(time.perf_counter() - started) * 1000,
                attempts=1,
                error=error,
            )
        )
        return state

    @staticmethod
    def _require_entities(state: StoryState) -> ExtractedEntities:
        """Вернуть реестр сущностей или понятную ошибку конфигурации."""

        if state.entities is None:
            raise PermanentImageGenerationError(
                "Нельзя иллюстрировать сказку: реестр сущностей не заполнен."
            )
        return state.entities

    @staticmethod
    def _resolve_output_dir(state: StoryState) -> Path:
        """Вернуть каталог вывода; он задаётся оркестратором до этапа."""

        if state.output_dir is None:
            raise PermanentImageGenerationError(
                "Нельзя сохранить иллюстрации: каталог вывода не задан."
            )
        return Path(state.output_dir)


__all__ = [
    "CHARACTER_SHEET_FILENAME",
    "ImageGeneratorStage",
    "page_filename",
]
