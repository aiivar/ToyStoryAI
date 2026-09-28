"""Доказательства удаления видеофлоу и мёртвого кода (задачи 7.1–7.4).

Видео выведено из продукта, поэтому в репозитории не должно остаться ни
модулей, ни настроек, ни методов репозитория, ни колонок, о которых код
знает. Тесты фиксируют именно это: они падают, пока удаляемое ещё лежит в
проекте, и становятся зелёными после уборки (D12).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Модули прежнего флоу: импорт обязан падать, а не молча вернуть заглушку.
REMOVED_MODULES = (
    "app.services.video_provider",
    "app.services.text_generator",
)


# --- 7.1: модуль видеопровайдера удалён ---------------------------------------


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_service_modules_are_gone(module_name: str) -> None:
    """Модулей прежнего флоу в проекте больше нет."""

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)

    assert not (PROJECT_ROOT / f"{module_name.replace('.', '/')}.py").exists()


def test_video_provider_files_are_deleted() -> None:
    """Файлы видеопровайдера и его тестов удалены из репозитория."""

    assert not (PROJECT_ROOT / "app/services/video_provider.py").exists()
    assert not (PROJECT_ROOT / "tests/test_video_provider.py").exists()


def test_bot_services_have_no_video_provider() -> None:
    """`BotServices` не собирает видеопровайдер (его нечем заменить)."""

    from app.bot.handlers import BotServices

    assert "video_provider" not in BotServices.__dataclass_fields__


# --- 7.2: настройки Kling и Gemini удалены -----------------------------------


def test_bot_settings_have_no_video_fields() -> None:
    """У `Settings` не осталось ни одного поля видеопровайдера."""

    from app.config import Settings

    fields = Settings.model_fields

    assert not [name for name in fields if name.startswith("kling_")]
    assert "mock_video_url" not in fields


def test_story_settings_have_no_gemini_fields() -> None:
    """У `StorySettings` не осталось полей Gemini."""

    from app.story_pipeline.settings import StorySettings

    assert not [
        name for name in StorySettings.model_fields if name.startswith("gemini_")
    ]


@pytest.mark.parametrize(
    "doc_name", (".env.example", "README.md"), ids=("env-example", "readme")
)
def test_docs_do_not_mention_kling_or_video_provider(doc_name: str) -> None:
    """В `.env.example` и README нет ни `KLING_*`, ни переменных видео."""

    text = (PROJECT_ROOT / doc_name).read_text(encoding="utf-8")

    assert "KLING_" not in text
    assert "MOCK_VIDEO_URL" not in text


# --- 7.3: пользовательские тексты без видео ----------------------------------


def test_user_facing_texts_do_not_mention_video() -> None:
    """Ни один пользовательский текст бота не упоминает видео.

    Проверяются все строки, экспортируемые модулями бота: забытая строка
    в `__all__` тоже попадёт в выборку.
    """

    from app.bot import handlers, runner

    for module in (handlers, runner):
        exported = getattr(module, "__all__", None) or [
            name
            for name in vars(module)
            if name.isupper() and isinstance(getattr(module, name), str)
        ]
        for name in exported:
            value = getattr(module, name)
            if not isinstance(value, str):
                continue
            assert "видео" not in value.lower(), f"{module.__name__}.{name}"


# --- 7.4: мёртвый код удалён -------------------------------------------------


def test_image_provider_has_no_gemini() -> None:
    """`GeminiImageProvider` удалён: иллюстрации рисует RouterAI."""

    from app.story_pipeline import providers

    image_module = importlib.import_module(
        "app.story_pipeline.providers.image"
    )

    assert not hasattr(image_module, "GeminiImageProvider")
    assert "GeminiImageProvider" not in image_module.__all__
    assert "GeminiImageProvider" not in providers.__dict__.get("__all__", [])


def test_photo_storage_has_no_encode_base64() -> None:
    """`encode_base64` удалён: base64 больше не нужен никому."""

    from app.services.storage import PhotoStorage

    assert not hasattr(PhotoStorage, "encode_base64")


def test_generation_type_has_no_video() -> None:
    """`GenerationType.VIDEO` удалён: артефакт у прогона один."""

    from app.db.models import GenerationType

    assert [item.value for item in GenerationType] == ["tale"]


def test_repository_has_no_video_api() -> None:
    """Репозиторий не умеет ни создавать видеозаписи, ни писать видеоссылки."""

    import inspect

    from app.db.repository import GenerationRepository

    assert not hasattr(GenerationRepository, "create_video")
    assert "video_url" not in inspect.signature(
        GenerationRepository.update_result
    ).parameters


def test_model_does_not_map_video_columns() -> None:
    """Модель не обращается к колонкам прежнего видеофлоу.

    Сами колонки в уже созданной `bot.db` остаются: `ALTER TABLE DROP COLUMN`
    не выполняется, они безвредны и исчезают пересозданием базы.
    """

    from app.db.models import Generation

    names = {column.name for column in Generation.__table__.columns}

    assert "video_url" not in names
    assert "kling_request_id" not in names


def test_sources_do_not_mention_removed_symbols() -> None:
    """В `app/` и `scripts/` не осталось упоминаний удалённых символов.

    Сканируются только файлы Python: `.env.example` и README проверяются
    отдельно, а `.pyc` в `__pycache__` — сборка, а не код.
    """

    removed = (
        "video_provider",
        "VideoProvider",
        "KlingProvider",
        "MockVideoProvider",
        "kling_api_key",
        "kling_request_id",
        "TextGenerator",
        "encode_base64",
        "GeminiImageProvider",
        "gemini_api_key",
        "GenerationType.VIDEO",
        "create_video",
    )
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "app").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders += [
            f"{path.relative_to(PROJECT_ROOT)}: {token}"
            for token in removed
            if token in text
        ]
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders += [
            f"{path.relative_to(PROJECT_ROOT)}: {token}"
            for token in removed
            if token in text
        ]

    assert not offenders, offenders
