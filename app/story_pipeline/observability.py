"""Наблюдаемость этапов пайплайна: единый формат лога и агрегация токенов.

Модуль размещён в корне подпакета, а не в `stages/`: логирует оркестратор,
и хелпер нужен именно ему. `log_stage` — тонкая обёртка над `loguru`: он не
знает ни состояния, ни провайдеров, поэтому в лог физически не могут попасть
ключи и токены — только имя этапа, краткая сводка, время и агрегированный
расход токенов.
"""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from app.story_pipeline.models import StageMetrics
from app.story_pipeline.providers.llm import Usage


def log_stage(
    stage: str,
    summary_in: str,
    summary_out: str,
    duration_ms: float,
    usage: Usage | None = None,
) -> None:
    """Записать строку наблюдаемости по завершённому этапу.

    `summary_in`/`summary_out` — краткие сводки состояния до и после этапа
    без промптов и секретов; `usage` — расход токенов LLM-этапа.
    """

    logger.info(
        "stage={} in=[{}] out=[{}] duration_ms={:.1f} tokens={}",
        stage,
        summary_in,
        summary_out,
        duration_ms,
        _format_usage(usage),
    )


def total_usage(metrics: list[StageMetrics]) -> Usage | None:
    """Сложить расход токенов этапов, сохранив «неизвестно» как `None`."""

    prompt_tokens = _sum_tokens(
        metric.prompt_tokens for metric in metrics
    )
    completion_tokens = _sum_tokens(
        metric.completion_tokens for metric in metrics
    )
    if prompt_tokens is None and completion_tokens is None:
        return None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _sum_tokens(values: Iterable[int | None]) -> int | None:
    """Просуммировать известные значения, не затирая итог значением `None`."""

    total: int | None = None
    for value in values:
        if value is None:
            continue
        total = (total or 0) + value
    return total


def _format_usage(usage: Usage | None) -> str:
    """Отрендерить расход токенов для лога; `-` означает «неизвестно»."""

    if usage is None:
        return "-"
    return f"{usage.prompt_tokens or 0}+{usage.completion_tokens or 0}"


__all__ = ["log_stage", "total_usage"]
