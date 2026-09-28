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


# --- Задача 2.4: запрос квоты -----------------------------------------------


async def test_quota_used_when_tale_and_video_succeed(
    repository: GenerationRepository,
) -> None:
    """Квота израсходована: есть и успешная сказка, и успешное видео."""

    tale = await repository.create_tale(505)
    video = await repository.create_video(505, request_id="req-3")
    await repository.update_result(
        tale.id,
        status=GenerationStatus.SUCCESS,
        tale_text="text",
    )
    await repository.update_result(
        video.id,
        status=GenerationStatus.SUCCESS,
        video_url="https://cdn/v.mp4",
    )

    assert await repository.has_completed_full_generation(505) is True


async def test_quota_not_used_with_only_tale(
    repository: GenerationRepository,
) -> None:
    """Квота не израсходована, пока нет успешного видео."""

    tale = await repository.create_tale(606)
    await repository.update_result(
        tale.id,
        status=GenerationStatus.SUCCESS,
        tale_text="text",
    )

    assert await repository.has_completed_full_generation(606) is False


async def test_quota_not_used_for_new_user(
    repository: GenerationRepository,
) -> None:
    """У нового пользователя квота не израсходована."""

    assert await repository.has_completed_full_generation(707) is False


# --- Квота «одна завершённая сказка» (has_completed_tale) --------------------


async def test_completed_tale_counts_as_used_story(
    repository: GenerationRepository,
) -> None:
    """Успешная сказка — использованная бесплатная сказка."""

    assert await repository.has_completed_tale(808) is False

    tale = await repository.create_tale(808)
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="text"
    )

    assert await repository.has_completed_tale(808) is True


async def test_old_style_tale_with_video_counts_as_used_story(
    repository: GenerationRepository,
) -> None:
    """Старая успешная сказка прежнего формата тоже засчитывается."""

    tale = await repository.create_tale(909)
    video = await repository.create_video(909, request_id="old")
    await repository.update_result(
        tale.id, status=GenerationStatus.SUCCESS, tale_text="text"
    )
    await repository.update_result(
        video.id, status=GenerationStatus.SUCCESS, video_url="https://cdn/v.mp4"
    )

    assert await repository.has_completed_tale(909) is True


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

    assert await repository.has_completed_tale(1010) is False


async def test_in_progress_run_does_not_consume_story(
    repository: GenerationRepository,
) -> None:
    """Идущий прогон квоту ещё не занимает: сказки пока нет."""

    await repository.create_tale(1111)

    assert await repository.has_completed_tale(1111) is False


async def test_story_of_another_user_does_not_count(
    repository: GenerationRepository,
) -> None:
    """Чужая успешная сказка не влияет на квоту пользователя."""

    other = await repository.create_tale(1212)
    await repository.update_result(
        other.id, status=GenerationStatus.SUCCESS, tale_text="text"
    )

    assert await repository.has_completed_tale(1313) is False


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

    assert await repository.has_completed_full_generation(808) is False


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
