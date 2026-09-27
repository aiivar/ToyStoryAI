"""Тесты настроек пайплайна (`app/story_pipeline/settings.py`)."""

from __future__ import annotations

from app.story_pipeline.settings import StorySettings, get_story_settings

# Переменные, которые читает StorySettings; в тестах изолируем окружение,
# чтобы значения из внешней среды не влияли на проверки по умолчанию.
_STORY_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT",
    "DEEPSEEK_MAX_TOKENS",
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
    "GEMINI_MODEL",
    "GEMINI_TIMEOUT",
    "STORY_DEFAULT_PAGES",
    "STORY_PAGES_MIN",
    "STORY_PAGES_MAX",
    "STORY_MAX_REVISIONS",
    "STORY_AGE_MIN",
    "STORY_AGE_MAX",
    "STORY_OUTPUT_DIR",
)


def _clear_story_env(monkeypatch) -> None:
    for name in _STORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # MAX-токен пайплайну не нужен; убираем, чтобы доказать независимость.
    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)


def test_defaults_without_max_bot_token(monkeypatch) -> None:
    """StorySettings создаётся без MAX_BOT_TOKEN и с ожидаемыми значениями."""

    _clear_story_env(monkeypatch)

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.deepseek_api_key == ""
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-flash"
    assert config.deepseek_timeout == 60.0
    assert config.deepseek_max_tokens == 4096

    assert config.gemini_api_key == ""
    assert config.gemini_base_url == "https://generativelanguage.googleapis.com"
    assert config.gemini_model == "gemini-3.1-flash-image"
    assert config.gemini_timeout == 120.0

    assert config.story_default_pages == 6
    assert config.story_pages_min == 1
    assert config.story_pages_max == 12
    assert config.story_max_revisions == 2
    assert config.story_age_min == 3
    assert config.story_age_max == 6
    assert config.story_output_dir == "output"


def test_environment_overrides_defaults(monkeypatch) -> None:
    """Значения из окружения переопределяют значения по умолчанию."""

    _clear_story_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-image")
    monkeypatch.setenv("STORY_DEFAULT_PAGES", "8")
    monkeypatch.setenv("STORY_AGE_MAX", "8")
    monkeypatch.setenv("STORY_OUTPUT_DIR", "runs")

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.deepseek_model == "deepseek-v4-pro"
    assert config.gemini_model == "gemini-2.5-flash-image"
    assert config.story_default_pages == 8
    assert config.story_age_max == 8
    assert config.story_output_dir == "runs"


def test_get_story_settings_returns_instance(monkeypatch, tmp_path) -> None:
    """Фабрика get_story_settings возвращает настроенный StorySettings."""

    _clear_story_env(monkeypatch)
    monkeypatch.setenv("MAX_BOT_TOKEN", "irrelevant-for-pipeline")
    # Изолируем от возможного локального .env в корне репозитория.
    monkeypatch.chdir(tmp_path)

    config = get_story_settings()

    assert isinstance(config, StorySettings)
    # Поле бота MAX не входит в настройки пайплайна.
    assert not hasattr(config, "max_bot_token")
    assert config.story_default_pages == 6


def test_settings_survive_example_env_file(monkeypatch, tmp_path) -> None:
    """Полный пример .env читается без обязательных секретов.

    CLI появится позже; здесь проверяем, что настройки пайплайна
    загружаются из .env, где новые переменные заполнены, а ключи пусты.
    """

    _clear_story_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MAX_BOT_TOKEN=\n"
        "DEEPSEEK_API_KEY=\n"
        "DEEPSEEK_MODEL=deepseek-flash\n"
        "GEMINI_API_KEY=\n"
        "GEMINI_MODEL=gemini-3.1-flash-image\n"
        "STORY_DEFAULT_PAGES=6\n"
        "STORY_OUTPUT_DIR=output\n",
        encoding="utf-8",
    )

    config = StorySettings(_env_file=str(env_file))  # type: ignore[call-arg]

    assert config.gemini_api_key == ""
    assert config.gemini_model == "gemini-3.1-flash-image"
    assert config.story_default_pages == 6
    assert config.story_output_dir == "output"
