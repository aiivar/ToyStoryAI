"""Тесты слоя хранения: модель, схема и репозиторий."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa

from app.db.engine import Database
from app.db.models import Generation, GenerationStatus, GenerationType
from app.db.repository import GenerationRepository

TMP_ROOT = Path(".tmp/tests")

# Текст сказки прежнего формата: 150–300 слов одним абзацем, тип `tale`.
# Именно такие записи лежат в боевой `bot.db` и должны засчитываться в квоту.
OLD_TALE_TEXT = " ".join(["Жил-был зайчик."] * 120)


# --- Задачи 2.1 и 2.2: модель и создание схемы ------------------------------


async def test_schema_creates_table_for_default_sqlite_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Таблица `generations` создаётся для URL из конфигурации по умолчанию."""

    workdir = (TMP_ROOT / f"schema-{uuid4().hex}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)

    db = Database("sqlite+aiosqlite:///./bot.db")
    try:
        await db.create_schema()
        assert (workdir / "bot.db").exists()

        async with db.engine.connect() as connection:
            names = await connection.run_sync(
                lambda sync_conn: sa.inspect(sync_conn).get_table_names()
            )
        assert "generations" in names
    finally:
        await db.dispose()


async def test_new_database_initialises_without_errors() -> None:
    """Новый файл базы инициализируется без ошибок."""

    db_path = TMP_ROOT / f"init-{uuid4().hex}.db"
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    try:
        await db.create_schema()
        assert db_path.exists()
    finally:
        await db.dispose()
        db_path.unlink(missing_ok=True)


# --- Задача 2.3: операции репозитория ---------------------------------------


async def test_create_tale_record(
    repository: GenerationRepository,
) -> None:
    """Запись о сказке создаётся со статусом «выполняется»."""

    record = await repository.create_tale(101, local_path="uploads/a.jpg")

    assert record.id is not None
    assert record.user_id == 101
    assert record.type is GenerationType.TALE
    assert record.status is GenerationStatus.IN_PROGRESS
    assert record.local_path == "uploads/a.jpg"
    assert record.created_at is not None


async def test_create_video_record_with_request_id(
    repository: GenerationRepository,
) -> None:
    """Запись о видео сохраняет идентификатор запроса провайдера."""

    record = await repository.create_video(
        101,
        request_id="req-abc",
        local_path="uploads/a.jpg",
    )

    assert record.type is GenerationType.VIDEO
    assert record.status is GenerationStatus.IN_PROGRESS
    assert record.kling_request_id == "req-abc"


async def test_update_status_and_result(
    repository: GenerationRepository,
) -> None:
    """Обновление переводит запись в терминальное состояние с результатом."""

    tale = await repository.create_tale(101)
    video = await repository.create_video(101, request_id="req-1")

    updated_tale = await repository.update_result(
        tale.id,
        status=GenerationStatus.SUCCESS,
        tale_text="Жила-была игрушка...",
    )
    updated_video = await repository.update_result(
        video.id,
        status=GenerationStatus.SUCCESS,
        video_url="https://cdn.example.com/v.mp4",
    )

    assert updated_tale is not None
    assert updated_tale.status is GenerationStatus.SUCCESS
    assert updated_tale.tale_text == "Жила-была игрушка..."
    assert updated_video is not None
    assert updated_video.status is GenerationStatus.SUCCESS
    assert updated_video.video_url == "https://cdn.example.com/v.mp4"


async def test_update_unknown_record_returns_none(
    repository: GenerationRepository,
) -> None:
    """Обновление несуществующей записи возвращает None."""

    result = await repository.update_result(
        999_999,
        status=GenerationStatus.FAILED,
        error="boom",
    )
    assert result is None


async def test_history_is_sorted_newest_first(
    repository: GenerationRepository,
) -> None:
    """История пользователя упорядочена от новых записей к старым."""

    first = await repository.create_tale(202)
    second = await repository.create_video(202, request_id="req-2")

    history = await repository.get_history(202)

    assert [item.id for item in history] == [second.id, first.id]


async def test_history_is_isolated_per_user(
    repository: GenerationRepository,
) -> None:
    """История одного пользователя не содержит записи других."""

    await repository.create_tale(303)
    foreign = await repository.create_tale(404)

    history = await repository.get_history(303)

    assert [item.user_id for item in history] == [303]
    assert foreign.id not in {item.id for item in history}


# --- Квота «одна завершённая сказка» (задачи 2.4 и 6.2) --------------------


async def test_completed_tale_counts_as_used_story(
    repository: GenerationRepository,
) -> None:
    """Успешная сказка — использованная бесплатная сказка."""

    assert await repository.has_completed_story(808) is False

    tale = await repository.create_tale(808)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="text"
    )

    assert await repository.has_completed_story(808) is True


async def test_old_format_tale_with_video_counts_as_used_story(
    repository: GenerationRepository,
) -> None:
    """Старая успешная сказка прежнего формата тоже засчитывается (D19).

    У прежних записей был ещё и успешный видеопрогон: без этого теста легко
    было бы сузить предикат до нового формата и отдать бесплатную книжку
    пользователю, который уже получил сказку.
    """

    tale = await repository.create_tale(909)
    video = await repository.create_video(909, request_id="old")
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text=OLD_TALE_TEXT
    )
    await repository.update_result(
        video.id, status=GenerationStatus.SUCCESS, video_url="https://cdn/v.mp4"
    )

    assert await repository.has_completed_story(909) is True


async def test_old_format_tale_without_video_counts_as_used_story(
    repository: GenerationRepository,
) -> None:
    """Старая успешная сказка засчитывается и без успешного видео."""

    tale = await repository.create_tale(919)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text=OLD_TALE_TEXT
    )

    assert await repository.has_completed_story(919) is True


async def test_failed_or_partial_run_keeps_story_available(
    repository: GenerationRepository,
) -> None:
    """Ошибка и частичный результат (статус FAILED) квоту не расходуют."""

    failed = await repository.create_tale(1010)
    await repository.update_result(
        failed.id, status=GenerationStatus.FAILED, error="boom"
    )
    partial = await repository.create_tale(1010)
    await repository.update_result(
        partial.id,
        status=GenerationStatus.FAILED,
        tale_text="часть страниц",
        error="нет картинок",
    )

    assert await repository.has_completed_story(1010) is False


async def test_in_progress_run_does_not_consume_story(
    repository: GenerationRepository,
) -> None:
    """Идущий прогон квоту ещё не занимает: сказки пока нет."""

    await repository.create_tale(1111)

    assert await repository.has_completed_story(1111) is False


async def test_story_of_another_user_does_not_count(
    repository: GenerationRepository,
) -> None:
    """Чужая успешная сказка не влияет на квоту пользователя."""

    other = await repository.create_tale(1212)
    await repository.update_result(
        other.id, status=GenerationStatus.SUCCESS, tale_text="text"
    )

    assert await repository.has_completed_story(1313) is False


async def test_quota_not_used_with_only_failures(
    repository: GenerationRepository,
) -> None:
    """Неудачные генерации не расходуют бесплатную квоту."""

    tale = await repository.create_tale(808)
    video = await repository.create_video(808, request_id="req-4")
    await repository.update_result(
        tale.id,
        status=GenerationStatus.FAILED,
        error="deepseek down",
    )
    await repository.update_result(
        video.id,
        status=GenerationStatus.FAILED,
        error="kling timeout",
    )

    assert await repository.has_completed_story(808) is False


async def test_video_success_alone_does_not_consume_story(
    repository: GenerationRepository,
) -> None:
    """Одного успешного видеопрогона для квоты недостаточно."""

    video = await repository.create_video(818, request_id="req-5")
    await repository.update_result(
        video.id, status=GenerationStatus.SUCCESS, video_url="https://cdn/v.mp4"
    )

    assert await repository.has_completed_story(818) is False


def test_dead_quota_method_is_gone() -> None:
    """Мёртвый предикат про «сказка + видео» удалён из репозитория."""

    assert not hasattr(GenerationRepository, "has_completed_full_generation")
    assert hasattr(GenerationRepository, "has_completed_story")


# --- Задача 6.4: содержимое записи -------------------------------------------


async def test_update_result_stores_pages_count_and_output_dir(
    repository: GenerationRepository,
) -> None:
    """Репозиторий сохраняет число страниц и каталог прогона."""

    tale = await repository.create_tale(1414)

    updated = await repository.update_result(
        tale.id,
        status=GenerationStatus.SUCCESS,
        tale_text="Одна страница. Другая страница.",
        pages_count=2,
        output_dir="output/1414-20260101-000000-000000",
    )

    assert updated is not None
    assert updated.pages_count == 2
    assert updated.output_dir == "output/1414-20260101-000000-000000"
    assert updated.tale_text == "Одна страница. Другая страница."


async def test_partial_run_keeps_pages_but_not_quota(
    repository: GenerationRepository,
) -> None:
    """Частичный прогон: страницы в записи есть, квота не израсходована."""

    tale = await repository.create_tale(1515)

    updated = await repository.update_result(
        tale.id,
        status=GenerationStatus.FAILED,
        tale_text="Страница 1. Страница 2.",
        pages_count=2,
        output_dir="output/1515-20260101-000000-000000",
        error="сервис не ответил",
    )

    assert updated is not None
    assert updated.pages_count == 2
    assert updated.status is GenerationStatus.FAILED
    assert await repository.has_completed_story(1515) is False


async def test_omitted_fields_keep_previous_values(
    repository: GenerationRepository,
) -> None:
    """Непереданные поля не затирают уже записанные значения."""

    tale = await repository.create_tale(1616)
    await repository.update_result(
        tale.id,
        status=GenerationStatus.SUCCESS,
        tale_text="текст",
        pages_count=4,
        output_dir="output/1616",
    )

    updated = await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, error=None
    )

    assert updated is not None
    assert updated.pages_count == 4
    assert updated.output_dir == "output/1616"
    assert updated.tale_text == "текст"


async def test_model_repr_mentions_identity() -> None:
    """Отладочное представление содержит ключевые поля."""

    record = Generation(
        user_id=1,
        type=GenerationType.TALE,
        status=GenerationStatus.IN_PROGRESS,
    )
    text = repr(record)
    assert "Generation" in text
    assert "user_id=1" in text
