"""Тесты настроек пайплайна (`app/story_pipeline/settings.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.story_pipeline.settings import (
    SUPPORTED_ASPECT_RATIOS,
    StorySettings,
    get_story_settings,
)

# Переменные, которые читает StorySettings; в тестах изолируем окружение,
# чтобы значения из внешней среды не влияли на проверки по умолчанию.
_STORY_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT",
    "DEEPSEEK_MAX_TOKENS",
    "ROUTERAI_API_KEY",
    "ROUTERAI_BASE_URL",
    "ROUTERAI_IMAGE_MODEL",
    "ROUTERAI_TIMEOUT",
    "STORY_IMAGE_ASPECT_RATIO",
    "STORY_DEFAULT_PAGES",
    "STORY_PAGES_MIN",
    "STORY_PAGES_MAX",
    "STORY_ILLUSTRATED_PAGES",
    "STORY_PAGE_WORDS_MIN",
    "STORY_PAGE_WORDS_MAX",
    "STORY_MAX_REVISIONS",
    "STORY_NARRATOR",
    "STORY_AGE_MIN",
    "STORY_AGE_MAX",
    "STORY_OUTPUT_DIR",
)


def _clear_story_env(monkeypatch) -> None:
    for name in _STORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # MAX-токен пайплайну не нужен; убираем, чтобы доказать независимость.
    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)


def _default_literal(value: object) -> str:
    """Запись дефолта так, как её пишут в `.env` и README.

    Целые таймауты в документации записаны без дробной части (`60`), хотя в
    модели это float, — иначе документ учил бы писать `60.0` там, где
    человек всё равно пишет `60`. Сравнение по точному `str()` ловило бы эту
    разницу записи, а не расхождение значения.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_env_example() -> str:
    """Текст `.env.example` — источник истины по именам переменных."""

    return (_PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")


def _read_readme() -> str:
    """Текст README с таблицей переменных окружения."""

    return (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_defaults_without_max_bot_token(monkeypatch) -> None:
    """StorySettings создаётся без MAX_BOT_TOKEN и с ожидаемыми значениями."""

    _clear_story_env(monkeypatch)

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.deepseek_api_key == ""
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-flash"
    assert config.deepseek_timeout == 60.0
    # Лимит поднят вдвое: при втрое более длинном тексте страниц и
    # STORY_PAGES_MAX = 12 прежних 4096 хватало впритык.
    assert config.deepseek_max_tokens == 8192

    assert config.routerai_api_key == ""
    assert config.routerai_base_url == "https://routerai.ru/api/v1"
    assert config.routerai_image_model == "google/gemini-3.1-flash-lite-image"
    assert config.routerai_timeout == 120.0

    assert config.story_image_aspect_ratio == "4:3"

    assert config.story_default_pages == 6
    assert config.story_pages_min == 1
    assert config.story_pages_max == 12
    assert config.story_illustrated_pages == 3
    # Границы объёма страницы: примерно втрое больше прежних 2–3 предложений,
    # но заданы числом слов, которое код способен посчитать и проверить.
    assert config.story_page_words_min == 60
    assert config.story_page_words_max == 120
    assert config.story_max_revisions == 2
    assert config.story_narrator == "first_person"
    assert config.story_age_min == 3
    assert config.story_age_max == 6
    assert config.story_output_dir == "output"


def test_environment_overrides_defaults(monkeypatch) -> None:
    """Значения из окружения переопределяют значения по умолчанию."""

    _clear_story_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://routerai.test/api/v1")
    monkeypatch.setenv("ROUTERAI_IMAGE_MODEL", "google/gemini-3.1-flash-image")
    monkeypatch.setenv("STORY_IMAGE_ASPECT_RATIO", "3:2")
    monkeypatch.setenv("STORY_DEFAULT_PAGES", "8")
    monkeypatch.setenv("STORY_ILLUSTRATED_PAGES", "0")
    monkeypatch.setenv("STORY_PAGE_WORDS_MIN", "40")
    monkeypatch.setenv("STORY_PAGE_WORDS_MAX", "90")
    monkeypatch.setenv("STORY_AGE_MAX", "8")
    monkeypatch.setenv("STORY_NARRATOR", "third_person_limited")
    monkeypatch.setenv("STORY_OUTPUT_DIR", "runs")

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.deepseek_model == "deepseek-v4-pro"
    assert config.routerai_base_url == "https://routerai.test/api/v1"
    assert config.routerai_image_model == "google/gemini-3.1-flash-image"
    assert config.story_image_aspect_ratio == "3:2"
    assert config.story_default_pages == 8
    assert config.story_illustrated_pages == 0
    assert config.story_page_words_min == 40
    assert config.story_page_words_max == 90
    assert config.story_age_max == 8
    assert config.story_narrator == "third_person_limited"
    assert config.story_output_dir == "runs"


def test_page_words_defaults_come_from_settings(monkeypatch) -> None:
    """Дефолты границ объёма живут в настройках, а не в коде этапов.

    Подбор цифр идёт по итогам живых прогонов и обязан обходиться без
    правки кода, поэтому дефолт — одно число в одном месте.
    """

    _clear_story_env(monkeypatch)

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.story_page_words_min == 60
    assert config.story_page_words_max == 120
    assert config.story_page_words_max > config.story_page_words_min


@pytest.mark.parametrize(
    ("words_min", "words_max", "expected_var"),
    [
        (120, 60, "STORY_PAGE_WORDS_MAX"),
        (100, 50, "STORY_PAGE_WORDS_MAX"),
        (0, 120, "STORY_PAGE_WORDS_MIN"),
        (60, 0, "STORY_PAGE_WORDS_MAX"),
        (-10, 120, "STORY_PAGE_WORDS_MIN"),
        (60, -10, "STORY_PAGE_WORDS_MAX"),
    ],
)
def test_degenerate_page_words_are_rejected(
    words_min: int,
    words_max: int,
    expected_var: str,
) -> None:
    """Вырожденная пара объёма отклоняется с упоминанием переменной.

    Пустой или неположительный диапазон — это не «писать как получится»,
    а опечатка в `.env`: обнаружить её надо до первого платного вызова,
    иначе прогон потратит деньги на заведомо деградированный текст.
    """

    with pytest.raises(ValidationError) as failure:
        StorySettings(  # type: ignore[call-arg]
            _env_file=None,
            story_page_words_min=words_min,
            story_page_words_max=words_max,
        )

    message = str(failure.value)
    assert expected_var in message
    assert "STORY_PAGE_WORDS" in message


def test_page_words_range_from_environment_is_validated(monkeypatch) -> None:
    """Ошибка в `.env` ловится при загрузке настроек, а не в середине прогона."""

    _clear_story_env(monkeypatch)
    monkeypatch.setenv("STORY_PAGE_WORDS_MIN", "150")
    monkeypatch.setenv("STORY_PAGE_WORDS_MAX", "80")

    with pytest.raises(ValidationError) as failure:
        StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert "STORY_PAGE_WORDS_MAX" in str(failure.value)


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


def test_unsupported_aspect_ratio_is_rejected() -> None:
    """Опечатка в STORY_IMAGE_ASPECT_RATIO отклоняется с понятным текстом."""

    with pytest.raises(ValidationError) as failure:
        StorySettings(  # type: ignore[call-arg]
            _env_file=None, story_image_aspect_ratio="4:31"
        )

    message = str(failure.value)
    # В сообщении есть и ошибочное значение, и список допустимых.
    assert "4:31" in message
    assert "16:9" in message


@pytest.mark.parametrize("aspect_ratio", SUPPORTED_ASPECT_RATIOS)
def test_supported_aspect_ratio_is_accepted(aspect_ratio: str) -> None:
    """Каждое поддерживаемое соотношение сторон проходит валидацию."""

    config = StorySettings(  # type: ignore[call-arg]
        _env_file=None, story_image_aspect_ratio=aspect_ratio
    )

    assert config.story_image_aspect_ratio == aspect_ratio


def test_empty_aspect_ratio_means_provider_default() -> None:
    """Пустое соотношение допустимо: провайдеру не отправляется ничего."""

    config = StorySettings(  # type: ignore[call-arg]
        _env_file=None, story_image_aspect_ratio="  "
    )

    assert config.story_image_aspect_ratio == ""


def test_env_example_documents_every_pipeline_setting(monkeypatch) -> None:
    """`.env.example` перечисляет каждое поле `StorySettings` с его дефолтом.

    Расхождение документации с кодом опаснее отсутствия записи: человек
    подкручивает прогон по файлу-примеру, меняет несуществующую переменную и
    получает поведение, которого не ожидал.
    """

    _clear_story_env(monkeypatch)
    config = StorySettings(_env_file=None)  # type: ignore[call-arg]
    env_example = _read_env_example()

    mismatches: list[str] = []
    for name, field in StorySettings.model_fields.items():
        var = name.upper()
        if var not in env_example:
            mismatches.append(f"{var}: нет в .env.example")
            continue
        default = field.get_default(call_default_factory=True)
        if default is None or isinstance(default, (str, list, tuple)):
            continue
        if f"{var}={_default_literal(default)}" not in env_example:
            mismatches.append(
                f"{var}: дефолт {_default_literal(default)} != .env.example"
            )

    assert mismatches == [], "; ".join(mismatches)


def test_readme_table_documents_every_pipeline_setting(monkeypatch) -> None:
    """Таблица README содержит каждую настройку пайплайна и её дефолт.

    Таблица — то, что читают перед первым запуском с реальным ключом; её
    расхождение с кодом стоит дороже всего остального.
    """

    _clear_story_env(monkeypatch)
    readme = _read_readme()

    missing: list[str] = []
    for name, field in StorySettings.model_fields.items():
        var = name.upper()
        row = next(
            (
                line
                for line in readme.splitlines()
                if line.startswith(f"| `{var}`")
            ),
            None,
        )
        if row is None:
            missing.append(f"{var}: нет строки в таблице README")
            continue
        default = field.get_default(call_default_factory=True)
        if default is None or isinstance(default, (str, list, tuple)):
            continue
        if f"| `{_default_literal(default)}` |" not in row:
            missing.append(
                f"{var}: дефолт {_default_literal(default)} != таблица README"
            )

    assert missing == [], "; ".join(missing)


def test_negative_illustration_budget_is_rejected() -> None:
    """Отрицательный бюджет иллюстраций — опечатка, а не «рисовать всё».

    Без проверки формула набора для `K = -1` дала бы пустой цикл и пустой
    набор, то есть требование «набор непуст» нарушило бы само себя и
    поймало бы только аварийный сбой. Ловим опечатку здесь, до первого
    платного вызова, — тем же способом, каким отсекается неверное
    соотношение сторон.
    """

    with pytest.raises(ValidationError) as failure:
        StorySettings(  # type: ignore[call-arg]
            _env_file=None, story_illustrated_pages=-1
        )

    assert "story_illustrated_pages" in str(failure.value)


def test_zero_illustration_budget_is_allowed() -> None:
    """Нулевой бюджет допустим: это явное «иллюстрировать все страницы»."""

    config = StorySettings(  # type: ignore[call-arg]
        _env_file=None, story_illustrated_pages=0
    )

    assert config.story_illustrated_pages == 0


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
        "ROUTERAI_API_KEY=\n"
        "ROUTERAI_IMAGE_MODEL=google/gemini-3.1-flash-lite-image\n"
        "STORY_DEFAULT_PAGES=6\n"
        "STORY_OUTPUT_DIR=output\n",
        encoding="utf-8",
    )

    config = StorySettings(_env_file=str(env_file))  # type: ignore[call-arg]

    assert config.routerai_api_key == ""
    assert config.routerai_image_model == "google/gemini-3.1-flash-lite-image"
    assert config.story_default_pages == 6
    assert config.story_output_dir == "output"


def test_gemini_variables_are_ignored(monkeypatch) -> None:
    """Переменные Gemini больше не читаются и не ломают загрузку.

    В локальных `.env` разработчиков они ещё остаются; если бы настройки
    пайплайна продолжали их читать, удаление провайдера было бы незаметным
    только до первого прогона.
    """

    _clear_story_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "sentinel-from-legacy-env")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-legacy-model")

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert not [
        name for name in StorySettings.model_fields if name.startswith("gemini_")
    ]
    assert not hasattr(config, "gemini_api_key")


# --- Озвучка -----------------------------------------------------------------


def _narration_settings(**overrides: object) -> StorySettings:
    """Настройки озвучки без чтения реального `.env`."""

    return StorySettings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_narration_voice_name_matches_selection() -> None:
    """Женский и мужской выбор отображаются в настроенные имена голосов."""

    config = _narration_settings(
        routerai_narration_voice_female="FemaleVoice",
        routerai_narration_voice_male="MaleVoice",
    )

    assert config.narration_voice_name("female") == "FemaleVoice"
    assert config.narration_voice_name("male") == "MaleVoice"


def test_empty_narration_model_is_rejected() -> None:
    """Пустая модель синтеза отклоняется при загрузке настроек."""

    with pytest.raises(ValidationError):
        _narration_settings(routerai_narration_model="   ")


def test_empty_narration_voice_is_rejected() -> None:
    """Пустое имя голоса отклоняется при загрузке настроек."""

    with pytest.raises(ValidationError):
        _narration_settings(routerai_narration_voice_female="")


def test_negative_retries_and_zero_concurrency_are_rejected() -> None:
    """Отрицательные повторы и нулевой параллелизм — ошибка настроек."""

    with pytest.raises(ValidationError):
        _narration_settings(routerai_narration_retries=-1)
    with pytest.raises(ValidationError):
        _narration_settings(routerai_narration_concurrency=0)


# --- Рассказчик ---------------------------------------------------------------


def test_narrator_defaults_to_first_person() -> None:
    """Без настройки рассказчик — игрушка, как и до появления выбора."""

    config = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert config.story_narrator == "first_person"


def test_narrator_is_validated_at_load_time() -> None:
    """Опечатка в STORY_NARRATOR называется вместе с переменной."""

    with pytest.raises(ValidationError) as failure:
        StorySettings(  # type: ignore[call-arg]
            _env_file=None, story_narrator="child"
        )

    message = str(failure.value)
    assert "STORY_NARRATOR" in message
    assert "first_person" in message
    assert "third_person_limited" in message
