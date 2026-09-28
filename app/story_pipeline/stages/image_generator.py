"""Этап иллюстрирования: character sheet и страницы через провайдер.

Тонкий узел графа: собирает промпты чистыми функциями `prompt_builder`,
вызывает инъектированный `ImageProvider` и сохраняет результат в
`state.output_dir`. Сначала генерируется один reference-лист персонажей
(с фото игрушки как единственным референсом, если оно передано), затем
каждая страница — с листом как единственным reference. Ретраи живут внутри
провайдера; этап отвечает за порядок, файлы и мягкую деградацию.

Имя файла строится по типу, который вернул провайдер (`image/jpeg` ->
`.jpg`): конвертировать картинку без новых зависимостей нечем, поэтому
честное имя честнее расширения вразрез с содержимым файла.

Каждому вызову передаётся детерминированный seed (`page_seed` для страниц,
`CHARACTER_SHEET_SEED` для листа), а в конце сверяется фактическое покрытие
страниц с `state.num_pages`. Это страховка: прогон не должен отчитываться
`status == "ok"`, если файл какой-то страницы не написан.
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

# Seed character sheet: не совпадает ни с одной страницей и постоянен между
# прогонами, поэтому mock рисует для него стабильную картинку. Провайдеры,
# не поддерживающие seed (Gemini), вернут `None` — это честнее, чем выдуманная
# воспроизводимость: см. README, раздел про seed.
CHARACTER_SHEET_SEED = 900

# Смещение seed страниц: page_seed(n) = PAGE_SEED_BASE + n. Смещение нужно,
# чтобы seed листа персонажей и seed страниц не пересекались.
PAGE_SEED_BASE = 1000

# Соотношение сторон character sheet: всегда квадрат. Референсный лист —
# не страница книги, и его пропорции не должны задавать пропорции страниц.
CHARACTER_SHEET_ASPECT_RATIO = "1:1"


def page_filename(page_num: int, media_type: str | None = None) -> str:
    """Имя файла страницы по её номеру и типу файла, который вернул провайдер.

    Расширение берётся из `media_type`, а не задаётся жёстко: живая проверка
    RouterAI отдаёт JPEG, и файл с расширением `.png` внутри содержал бы
    JPEG-байты. Конвертация не выполняется (зависимостей добавлять нельзя),
    поэтому единственный честный выход — назвать файл по его типу.
    Пустой тип даёт `.png`; тип вне таблицы сохраняет свой подтип
    (`image/avif` → `.avif`).
    """

    return f"page_{page_num}{file_extension(media_type)}"


def character_sheet_filename(media_type: str | None = None) -> str:
    """Имя файла character sheet по типу файла, который вернул провайдер."""

    return f"character_sheet{file_extension(media_type)}"


# Расширение по умолчанию, когда тип файла узнать не удалось: PNG —
# историческое умолчание mock- и Gemini-провайдеров, поэтому прогон
# остаётся рабочим. Задано расширением, а не MIME-типом, потому что
# функция возвращает именно расширение, а тип из ответа провайдера при
# этом может отсутствовать или быть нечитаемым.
_DEFAULT_EXTENSION = ".png"

# Соответствие MIME-типов и расширений файлов. `image/jpeg` по RFC пишется
# и `.jpg`, и `.jpeg`; выбран короткий вариант, его понимают все клиенты.
# Типы вне таблицы (например, `image/avif`) получают подтип как есть.
_MEDIA_TYPE_EXTENSIONS: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def file_extension(media_type: str | None) -> str:
    """Расширение имени файла по MIME-типу; `.png`, если тип не узнать.

    Тип вне таблицы отдаёт свой подтип (`.avif` для `image/avif`), но
    подтип фильтруется: в имя попадают только буквы и цифры, поэтому тип от
    провайдера не может подсунуть путь (`../../etc/passwd`) или имя без
    расширения. Пустой или вовсе нечитаемый тип даёт `.png`.
    """

    normalized = (media_type or "").strip().lower()
    if not normalized:
        return _DEFAULT_EXTENSION
    if normalized in _MEDIA_TYPE_EXTENSIONS:
        return _MEDIA_TYPE_EXTENSIONS[normalized]
    subtype = normalized.rpartition("/")[2]
    safe = "".join(char for char in subtype if char.isalnum())
    if not safe:
        return _DEFAULT_EXTENSION
    return f".{safe}"


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

    def __init__(
        self,
        provider: ImageProvider,
        *,
        reference_photo: bytes | None = None,
        aspect_ratio: str | None = None,
    ) -> None:
        """Запомнить провайдера и входные данные прогона.

        `reference_photo` — байты фото игрушки: единственный референс для
        character sheet. Они передаются конструктором, а не состоянием,
        потому что `StoryState` сериализуется в `story.json` и бинарей не
        содержит. `aspect_ratio` — соотношение сторон страниц; `None`
        оставляет умолчание провайдера.
        """

        self._provider = provider
        self._reference_photo = reference_photo
        self._aspect_ratio = aspect_ratio

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

        # Фото игрушки — единственный референс листа; без фото лист строится
        # только по описанию (CLI-сценарий). Страницы ниже получают лист.
        sheet = await self._provider.generate(
            build_character_sheet_prompt(
                entities, toy_description=state.toy_description
            ),
            seed=CHARACTER_SHEET_SEED,
            reference_images=(
                [self._reference_photo]
                if self._reference_photo is not None
                else None
            ),
            aspect_ratio=CHARACTER_SHEET_ASPECT_RATIO,
        )
        sheet_path = output_dir / character_sheet_filename(sheet.media_type)
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
                    aspect_ratio=self._aspect_ratio,
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
            path = output_dir / page_filename(
                scene.page_num, image.media_type
            )
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

        Проверяются три вещи: есть ли файл для каждой страницы 1..N, не лишние
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
                "нет файлов страниц: "
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
    "CHARACTER_SHEET_ASPECT_RATIO",
    "CHARACTER_SHEET_SEED",
    "PAGE_SEED_BASE",
    "ImageGeneratorStage",
    "character_sheet_filename",
    "file_extension",
    "page_filename",
    "page_seed",
]
