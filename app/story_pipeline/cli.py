"""Командный интерфейс пайплайна сказок.

Запуск: ``python -m app.story_pipeline --theme ... --age ... --moral ...``.
CLI создаёт каталог ``output/{timestamp}/`` с иллюстрациями и полным
``story.json``. Для тестируемости точка входа `run_cli` принимает настройки
и провайдеры снаружи, а провайдеры, созданные самим CLI, закрываются им же.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import StoryState
from app.story_pipeline.pipeline import StoryPipeline
from app.story_pipeline.providers.image import (
    ImageProvider,
    make_image_provider,
)
from app.story_pipeline.providers.llm import DeepSeekLLMClient, LLMClient
from app.story_pipeline.settings import StorySettings, get_story_settings

# Код возврата при ошибках разбора и валидации аргументов.
_EXIT_USAGE = 2

# Код возврата при постоянной ошибке прогона.
_EXIT_FAILURE = 1

# Код возврата успешного прогона (в т.ч. `needs_review`).
_EXIT_OK = 0

# Формат метки каталога прогона: секундная точность, сортируется лексикографически.
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

# Имя файла полного состояния прогона.
_STORY_FILENAME = "story.json"


def run_cli(
    argv: Sequence[str],
    *,
    settings: StorySettings | None = None,
    llm: LLMClient | None = None,
    image_provider: ImageProvider | None = None,
    now: Callable[[], datetime] | None = None,
) -> int:
    """Разобрать аргументы и вернуть код возврата, не бросая наружу сбои."""

    resolved = settings if settings is not None else get_story_settings()
    clock = now if now is not None else _utcnow
    parser = _build_parser(resolved.story_default_pages)
    args = parser.parse_args(argv)

    if not (resolved.story_age_min <= args.age <= resolved.story_age_max):
        print(
            "Ошибка: возраст "
            f"{args.age} вне допустимого диапазона "
            f"{resolved.story_age_min}..{resolved.story_age_max}.",
            file=sys.stderr,
        )
        return _EXIT_USAGE

    if not (
        resolved.story_pages_min <= args.pages <= resolved.story_pages_max
    ):
        print(
            "Ошибка: число страниц "
            f"{args.pages} вне допустимого диапазона "
            f"{resolved.story_pages_min}..{resolved.story_pages_max}.",
            file=sys.stderr,
        )
        return _EXIT_USAGE

    run_dir = _resolve_run_dir(args.output, resolved, clock)
    run_settings = resolved.model_copy(
        update={"story_output_dir": str(run_dir)}
    )

    resolved_llm = llm if llm is not None else DeepSeekLLMClient(run_settings)
    resolved_image = (
        image_provider
        if image_provider is not None
        else make_image_provider(run_settings)
    )
    # CLI закрывает только те провайдеры, что создал сам; инъектированные
    # остаются собственностью вызывающего (тесты и внешние композиции).
    owned: list[object] = []
    if llm is None:
        owned.append(resolved_llm)
    if image_provider is None:
        owned.append(resolved_image)

    state = StoryState(
        theme=args.theme,
        age=args.age,
        moral=args.moral,
        num_pages=args.pages,
        started_at=clock(),
    )
    pipeline = StoryPipeline(resolved_llm, resolved_image, run_settings)

    try:
        state = asyncio.run(_execute(pipeline, state, owned))
    except StoryPipelineError as exc:
        print(
            f"Не удалось создать сказку: {exc}",
            file=sys.stderr,
        )
        return _EXIT_FAILURE

    _write_story_json(run_dir, state)
    print(f"Статус: {state.status}")
    print(f"Каталог результатов: {run_dir}")
    return _EXIT_OK


async def _execute(
    pipeline: StoryPipeline,
    state: StoryState,
    owned: list[object],
) -> StoryState:
    """Прогнать пайплайн и гарантированно закрыть созданные CLI провайдеры."""

    try:
        return await pipeline.run(state)
    finally:
        await _close_owned(owned)


async def _close_owned(providers: list[object]) -> None:
    """Закрыть провайдеры с `aclose`, не давая сбою закрытия сорвать прогон."""

    for provider in providers:
        close = getattr(provider, "aclose", None)
        if not callable(close):
            continue
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - закрыть и остальные
            logger.warning("story cli: failed to close a provider: {}", exc)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI: аргументы берутся из `sys.argv` при вызове модуля."""

    return run_cli(sys.argv[1:] if argv is None else argv)


def _build_parser(default_pages: int) -> argparse.ArgumentParser:
    """Собрать парсер аргументов CLI с русскими подсказками."""

    parser = argparse.ArgumentParser(
        prog="python -m app.story_pipeline",
        description="Сгенерировать персональную сказку с иллюстрациями.",
    )
    parser.add_argument("--theme", required=True, help="Тема сказки.")
    parser.add_argument(
        "--age",
        type=int,
        required=True,
        help="Целевой возраст читателя.",
    )
    parser.add_argument("--moral", required=True, help="Мораль сказки.")
    parser.add_argument(
        "--pages",
        type=int,
        default=default_pages,
        help="Число страниц сказки.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Корневой каталог результатов (по умолчанию STORY_OUTPUT_DIR).",
    )
    return parser


def _resolve_run_dir(
    output: str | None,
    settings: StorySettings,
    now: Callable[[], datetime],
) -> Path:
    """Каталог прогона: корень из `--output`/настроек плюс метка времени."""

    root = (
        Path(output)
        if output is not None
        else Path(settings.story_output_dir)
    )
    return root / now().strftime(_TIMESTAMP_FORMAT)


def _write_story_json(run_dir: Path, state: StoryState) -> None:
    """Записать полное состояние прогона в ``story.json``.

    Каталог создаётся здесь: к моменту записи этап иллюстрирования уже
    наполнил его файлами иллюстраций, а `exist_ok` страхует будущие графы без
    иллюстраций. Так при сбое до записи пустой каталог не остаётся.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _STORY_FILENAME).write_text(
        json.dumps(
            state.model_dump(mode="json"), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def _utcnow() -> datetime:
    """Текущий момент в UTC — часы CLI по умолчанию."""

    return datetime.now(timezone.utc)


__all__ = ["main", "run_cli"]
