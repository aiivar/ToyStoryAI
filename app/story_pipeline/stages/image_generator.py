"""Этап иллюстрирования: character sheet и PNG-страницы через провайдер.

Тонкий узел графа: собирает промпты чистыми функциями `prompt_builder`,
вызывает инъецированный `ImageProvider` и сохраняет результат в
`state.output_dir`. Сначала генерируется один reference-лист персонажей,
затем каждая страница — с ним как reference. Ретраи живут внутри
провайдера; этап отвечает за порядок, файлы и мягкую деградацию.

Каждому вызову передаётся детерминированный seed (`page_seed` для страниц,
`CHARACTER_SHEET_SEED` для листа), а в конце сверяется фактическое покрытие
страниц с `state.num_pages`. Это страховка: прогон не должен отчитываться
`status == "ok"`, если PNG какой-то страницы не написан.
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

# Seed character sheet: не совпадает ни с одной страницей и постоянен между
# прогонами, поэтому mock рисует для него стабильную картинку. Провайдеры,
# не поддерживающие seed (Gemini), вернут `None` — это честнее, чем выдуманная
# воспроизводимость: см. README, раздел про seed.
CHARACTER_SHEET_SEED = 900

# Смещение seed страниц: page_seed(n) = PAGE_SEED_BASE + n. Смещение нужно,
# чтобы seed листа персонажей и seed страниц не пересекались.
PAGE_SEED_BASE = 1000


def page_filename(page_num: int) -> str:
    """Имя PNG-файла страницы по её номеру."""

    return f"page_{page_num}.png"


def page_seed(page_num: int) -> int:
    """Детерминированный seed страницы, выведенный из её номера.

    Разные страницы получают разные seed, одна и та же страница — одинаковый
    при повторном прогоне. В mock-режиме это делает страницы разными
    картинками; провайдер, поддерживающий seed, получает воспроизводимый
    запрос.
    """

    return PAGE_SEED_BASE + page_num


class ImageGeneratorStage:
    """Узел графа «Иллюстрирование»: sheet + страницы через `ImageProvider`."""

    name = "illustrate"

    def __init__(self, provider: ImageProvider) -> None:
        self._provider = provider

    async def run(self, state: StoryState) -> StoryState:
        """Сгенерировать reference-лист и страницы, записав пути и seed.

        В конце прогон проверяется на полноту: ожидаемые номера страниц
        сверяются с фактически записанными файлами. Любой пробел (сбой
        генерации, потерянная сцена, номер вне 1..N) переводит прогон в
        `needs_review` и попадает в `StageMetrics.error` — иначе результат с
        недостающей иллюстрацией выглядел бы успешным.
        """

        started = time.perf_counter()
        output_dir = self._resolve_output_dir(state)
        entities = self._require_entities(state)
        output_dir.mkdir(parents=True, exist_ok=True)

        sheet = await self._provider.generate(
            build_character_sheet_prompt(entities), seed=CHARACTER_SHEET_SEED
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
                    prompt,
                    seed=page_seed(scene.page_num),
                    reference_images=[sheet.data],
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

        error = self._coverage_error(state, failed_pages)
        if error is not None:
            state.status = "needs_review"
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
    def _coverage_error(
        state: StoryState,
        failed_pages: list[int],
    ) -> str | None:
        """Сверить ожидаемые страницы с записанными и описать расхождение.

        Проверяются три вещи: есть ли PNG для каждой страницы 1..N, не лишние
        ли это страницы и действительно ли файлы лежат на диске по указанным
        путям. Сбойные страницы провайдера добавляются отдельным пунктом.
        `None` означает полное покрытие.
        """

        expected = set(range(1, state.num_pages + 1))
        written = {art.page_num for art in state.images}
        missing = sorted(expected - written)
        out_of_range = sorted(written - expected)
        absent_files = sorted(
            art.page_num
            for art in state.images
            if not Path(art.path).is_file()
        )
        problems: list[str] = []
        if missing:
            problems.append(
                "нет PNG страниц: "
                + ", ".join(str(page_num) for page_num in missing)
            )
        if out_of_range:
            numbers = ", ".join(
                str(page_num) for page_num in out_of_range
            )
            problems.append(f"страницы вне 1..{state.num_pages}: {numbers}")
        if absent_files:
            problems.append(
                "файлы страниц не записаны: "
                + ", ".join(str(page_num) for page_num in absent_files)
            )
        if failed_pages:
            problems.append(
                "не сгенерированы страницы: "
                + ", ".join(str(page_num) for page_num in sorted(failed_pages))
            )
        return "; ".join(problems) if problems else None

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
    "CHARACTER_SHEET_SEED",
    "PAGE_SEED_BASE",
    "ImageGeneratorStage",
    "page_filename",
    "page_seed",
]
