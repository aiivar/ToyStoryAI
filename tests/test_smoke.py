"""Тесты smoke-скрипта: режимы без платных вызовов (задача 6.3).

Скрипт — ручной инструмент эксплуатации, поэтому проверяем не сеть, а
разбор режимов и тексты: платных вызовов в smoke быть не должно, а
отсутствие ключа RouterAI обязано быть видно сразу.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from scripts import smoke


def make_settings(**overrides: str) -> Settings:
    """Настройки для скрипта: без `.env`, с тестовым токеном MAX."""

    values: dict[str, object] = {
        "max_bot_token": "test-token",
        "routerai_api_key": "",
        "deepseek_api_key": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def test_image_mode_reports_mock_without_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Без `ROUTERAI_API_KEY` скрипт говорит про mock-картинки."""

    smoke.report_image_mode(make_settings())

    output = capsys.readouterr().out
    assert "ROUTERAI_API_KEY" in output
    assert "mock" in output.lower()


def test_image_mode_reports_real_provider_without_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """С ключом скрипт называет провайдера и не делает платный вызов.

    Любая попытка сети упала бы на фикстуре, запрещающей живые вызовы в
    unit-тестах, поэтому успешное выполнение функции — доказательство
    отсутствия запроса.
    """

    settings = make_settings(
        routerai_api_key="test-key", routerai_image_model="test/image-model"
    )

    smoke.report_image_mode(settings)

    output = capsys.readouterr().out
    assert "test/image-model" in output
    assert "mock" not in output.lower()


def test_smoke_has_no_video_report() -> None:
    """Отчёт о видеопровайдере из скрипта убран (D12)."""

    assert not hasattr(smoke, "report_video_mode")
    assert not hasattr(smoke, "MockVideoProvider")
    assert not hasattr(smoke, "build_video_provider")


def test_smoke_does_not_call_toy_vision() -> None:
    """Распознавание игрушки — платный вызов, в smoke его нет вовсе."""

    assert not hasattr(smoke, "check_toy_vision")
    assert not hasattr(smoke, "ToyVisionService")


def test_default_arguments_enable_no_paid_check() -> None:
    """Без флагов запускаются только бесплатные проверки."""

    args = smoke.parse_args([])

    assert args.skip_max is False
    assert args.skip_deepseek is False


def test_configuration_only_run_reports_being_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Прогон только с пропущенными сетевыми проверками ничего не платит."""

    monkeypatch.setenv("MAX_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("ROUTERAI_API_KEY", "")
    monkeypatch.setattr("sys.argv", ["smoke.py", "--skip-max", "--skip-deepseek"])

    assert asyncio.run(smoke.main()) == 0

    output = capsys.readouterr().out
    assert smoke.NO_PAID_CALLS_NOTICE in output
    assert "Видео" not in output
