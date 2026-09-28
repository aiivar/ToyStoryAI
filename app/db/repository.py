"""Репозиторий истории генераций.

Инкапсулирует операции записи и чтения записей `Generation`, включая
запрос квоты: получена ли пользователем бесплатная сказка.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import Database
from app.db.models import Generation, GenerationStatus, GenerationType


class GenerationRepository:
    """Операции над записями о генерациях."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_tale(
        self,
        user_id: int,
        *,
        status: GenerationStatus = GenerationStatus.IN_PROGRESS,
        local_path: str | None = None,
        tale_text: str | None = None,
    ) -> Generation:
        """Создать запись о сказке (обычно в начале попытки)."""

        return await self._create(
            user_id=user_id,
            generation_type=GenerationType.TALE,
            status=status,
            local_path=local_path,
            tale_text=tale_text,
        )

    async def _create(
        self,
        *,
        user_id: int,
        generation_type: GenerationType,
        status: GenerationStatus,
        local_path: str | None = None,
        tale_text: str | None = None,
    ) -> Generation:
        async with self._db.session_scope() as session:
            generation = Generation(
                user_id=user_id,
                type=generation_type,
                status=status,
                local_path=local_path,
                tale_text=tale_text,
            )
            session.add(generation)
            await session.commit()
            await session.refresh(generation)
            return generation

    async def update_result(
        self,
        generation_id: int,
        *,
        status: GenerationStatus,
        tale_text: str | None = None,
        error: str | None = None,
        pages_count: int | None = None,
        output_dir: str | None = None,
    ) -> Generation | None:
        """Обновить статус и результат записи.

        Значения, которые не переданы, остаются прежними: `None` означает
        «нечего менять», а не «очистить колонку».

        Возвращает обновлённую запись или `None`, если её нет.
        """

        async with self._db.session_scope() as session:
            generation = await session.get(Generation, generation_id)
            if generation is None:
                return None

            generation.status = status
            if tale_text is not None:
                generation.tale_text = tale_text
            if error is not None:
                generation.error = error
            if pages_count is not None:
                generation.pages_count = pages_count
            if output_dir is not None:
                generation.output_dir = output_dir

            await session.commit()
            await session.refresh(generation)
            return generation

    async def get_by_id(self, generation_id: int) -> Generation | None:
        """Вернуть запись по идентификатору."""

        async with self._db.session_scope() as session:
            return await session.get(Generation, generation_id)

    async def get_history(
        self,
        user_id: int,
        *,
        limit: int | None = None,
    ) -> Sequence[Generation]:
        """История сказок пользователя от новых записей к старым.

        Отбор по `type == TALE` сделан прямо в SQL не из вкуса, а по
        необходимости: в базах прежнего флоу остались строки `type = 'video'`,
        а `GenerationType` их больше не знает, и загрузка такой строки в ORM
        упала бы с `LookupError`. Такие строки относятся к удалённой
        функции, поэтому в истории сказок им не место.
        """

        statement = (
            select(Generation)
            .where(
                Generation.user_id == user_id,
                Generation.type == GenerationType.TALE,
            )
            .order_by(Generation.created_at.desc(), Generation.id.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)

        async with self._db.session_scope() as session:
            result = await session.execute(statement)
            return result.scalars().all()

    async def has_completed_story(self, user_id: int) -> bool:
        """Есть ли у пользователя завершённая сказка (бесплатная квота).

        Правило одно: одна завершённая сказка на пользователя (D11). Тип
        записи прежней версии — тоже `tale`, поэтому старые успешные
        сказки прежнего формата засчитываются автоматически (D19), и
        пользователь, уже получивший сказку, не получает книжку бесплатно.

        Частичный прогон и ошибка квоту не занимают: их статус `FAILED`.
        """

        statement = select(func.count()).select_from(Generation).where(
            Generation.user_id == user_id,
            Generation.type == GenerationType.TALE,
            Generation.status == GenerationStatus.SUCCESS,
        )

        async with self._db.session_scope() as session:
            result = await session.execute(statement)
            return bool(result.scalar())


__all__ = ["GenerationRepository"]
