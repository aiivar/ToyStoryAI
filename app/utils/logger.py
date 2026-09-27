"""Настройка логирования на базе loguru.

Единая точка конфигурации логов: уровень берётся из `Settings.log_level`,
вывод идёт в stderr в компактном структурированном формате. Сообщения
сторонних библиотек (maxapi, aiohttp) через стандартный `logging`
перенаправляются в loguru, чтобы был виден весь поток логов.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger

from app.config import Settings

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Логгеры сторонних библиотек, которые нужно перехватить.
_INTERCEPTED_LOGGERS = (
    "maxapi",
    "bot",
    "connection",
    "dispatcher",
    "aiohttp",
    "asyncio",
)


class _InterceptHandler(logging.Handler):
    """Перенаправляет записи стандартного `logging` в loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).log(
            level, "[{}] {}", record.name, record.getMessage()
        )


def setup_logging(settings: Settings) -> None:
    """Переконфигурировать loguru под уровень из настроек.

    Повторный вызов безопасен: предыдущие обработчики удаляются.
    """

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=_LOG_FORMAT,
        backtrace=False,
        diagnose=False,
    )

    intercept = _InterceptHandler()
    logging.basicConfig(
        handlers=[intercept],
        level=settings.log_level.upper(),
        force=True,
    )
    for name in _INTERCEPTED_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [intercept]
        stdlib_logger.propagate = False
        stdlib_logger.setLevel(settings.log_level.upper())


def get_logger():  # noqa: ANN201 - возвращаем объект loguru.logger
    """Вернуть общий логгер приложения."""

    return logger


__all__ = ["get_logger", "setup_logging"]
