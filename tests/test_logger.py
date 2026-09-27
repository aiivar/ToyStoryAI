"""Smoke-тесты логирования (`app/utils/logger.py`)."""

from __future__ import annotations

from app.config import Settings
from app.utils.logger import get_logger, setup_logging


def _settings(level: str) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
        log_level=level,
    )


def test_logs_at_configured_level(capsys) -> None:
    """При уровне INFO сообщения INFO выводятся, DEBUG — нет."""

    setup_logging(_settings("INFO"))
    logger = get_logger()

    logger.debug("debug-should-be-hidden")
    logger.info("info-should-be-visible")

    err = capsys.readouterr().err
    assert "info-should-be-visible" in err
    assert "debug-should-be-hidden" not in err


def test_debug_level_emits_debug(capsys) -> None:
    """При уровне DEBUG выводятся и отладочные сообщения."""

    setup_logging(_settings("DEBUG"))
    get_logger().debug("debug-visible-at-debug-level")

    err = capsys.readouterr().err
    assert "debug-visible-at-debug-level" in err
